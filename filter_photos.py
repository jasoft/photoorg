import os
import shutil
import cv2
import argparse
import json
import numpy as np
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from PIL import Image
from ultralytics import YOLO

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    print("Warning: pillow-heif not installed. HEIC support disabled.")

DELETE_DIR_NAME = "to_delete"
PROCESSED_DIR_NAME = "processed"
METADATA_FILE = "restore_map.json"
LOG_FILE = "scan.log"
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASSES = [0, 15, 16]

# Thread-local storage for non-thread-safe objects (like OpenCV CascadeClassifier)
thread_local_data = threading.local()


def setup_logging(input_dir):
    log_path = os.path.join(input_dir, LOG_FILE)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Also print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    return log_path


def setup_directories(base_path):
    delete_dir = os.path.join(base_path, DELETE_DIR_NAME)
    os.makedirs(delete_dir, exist_ok=True)
    return delete_dir


def load_restore_map(base_path):
    map_path = os.path.join(base_path, DELETE_DIR_NAME, METADATA_FILE)
    if os.path.exists(map_path):
        with open(map_path, "r") as f:
            return json.load(f)
    return {}


def save_restore_map(base_path, restore_map):
    map_path = os.path.join(base_path, DELETE_DIR_NAME, METADATA_FILE)
    with open(map_path, "w") as f:
        json.dump(restore_map, f, indent=2)


def load_image(filepath):
    """
    Load image using Pillow (handles HEIC) and convert to OpenCV BGR format.
    Returns: numpy array (BGR) or None
    """
    try:
        pil_image = Image.open(filepath)
        pil_image = pil_image.convert("RGB")
        img_rgb = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        logging.error(f"Error loading image {os.path.basename(filepath)}: {e}")
        return None


def has_face(image_rgb, person_bbox, face_cascade_path):
    # Retrieve or initialize thread-local CascadeClassifier
    if not hasattr(thread_local_data, "face_cascade"):
        thread_local_data.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    face_cascade = thread_local_data.face_cascade

    h, w, _ = image_rgb.shape
    x1, y1, x2, y2 = map(int, person_bbox)

    pad = 20
    y1_pad = max(0, y1 - pad)
    y2_pad = min(h, y2 + pad)
    x1_pad = max(0, x1 - pad)
    x2_pad = min(w, x2 + pad)

    crop = image_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0


def process_single_file(
    filepath, model, face_cascade_path, delete_dir, restore_map, map_lock
):
    filename = os.path.basename(filepath)
    try:
        logging.info(f"Processing: {filepath}")
        image_bgr = load_image(filepath)
        if image_bgr is None:
            return False

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Ultralytics YOLO inference is generally thread-safe, but if issues arise, use a lock for model
        results = model(image_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)

        found_valid_subject = False
        is_person_without_face = False
        keep_reason = ""

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])

                if cls in TARGET_CLASSES:
                    if cls == 0:  # Person
                        if has_face(image_rgb, box.xyxy[0], face_cascade_path):
                            found_valid_subject = True
                            keep_reason = "Person with face detected"
                            break
                        else:
                            is_person_without_face = True
                    else:  # Pet
                        found_valid_subject = True
                        cls_name = model.names[cls]
                        keep_reason = f"Pet ({cls_name}) detected"
                        break

            if found_valid_subject:
                break

        if found_valid_subject:
            logging.info(f"  [KEEP] {filename} - Reason: {keep_reason} (Left in place)")
            return True
        else:
            reason = (
                "No person/pet found"
                if not is_person_without_face
                else "Person detected but NO face visible"
            )
            logging.info(f"  [FILTER] {filename} - Reason: {reason}")

            map_key = filename
            dest_path = os.path.join(delete_dir, filename)

            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest_path):
                new_filename = f"{base}_{counter}{ext}"
                dest_path = os.path.join(delete_dir, new_filename)
                map_key = new_filename
                counter += 1

            # Critical section for shared resource
            with map_lock:
                restore_map[map_key] = filepath

            shutil.move(filepath, dest_path)
            return True

    except Exception as e:
        logging.error(f"  Error processing {filename}: {e}")
        return False


def process_photos(input_dir, restore_mode=False, workers=4):
    input_dir = os.path.abspath(input_dir)
    delete_dir = setup_directories(input_dir)

    # Setup logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_file_path = setup_logging(input_dir)

    restore_map = load_restore_map(input_dir)

    if restore_mode:
        logging.info(f"--- STARTING RESTORE MODE in {delete_dir} ---")
        restored_count = 0
        current_files = set(os.listdir(delete_dir))

        for filename in list(current_files):
            if filename == METADATA_FILE:
                continue

            if filename in restore_map:
                original_path = restore_map[filename]
                current_path = os.path.join(delete_dir, filename)

                if os.path.exists(current_path):
                    logging.info(f"Restoring: {filename} -> {original_path}")
                    os.makedirs(os.path.dirname(original_path), exist_ok=True)
                    shutil.move(current_path, original_path)
                    restored_count += 1

        logging.info(f"Restored {restored_count} files.")
        remaining_files = [f for f in os.listdir(delete_dir) if f != METADATA_FILE]
        new_map = {k: v for k, v in restore_map.items() if k in remaining_files}
        save_restore_map(input_dir, new_map)
        return

    logging.info(f"--- STARTING SCAN: {input_dir} (Workers: {workers}) ---")

    # Initialize expensive models once
    model = YOLO("yolov8n.pt")
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    # Shared lock for restore_map
    map_lock = threading.Lock()

    logging.info(f"Scanning files in {input_dir}...")

    processed_count = 0

    # Use ThreadPoolExecutor to process files as they are found (streaming)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = set()

        for root, dirs, files in os.walk(input_dir, topdown=True):
            # Prune specific directories to prevent traversal
            if DELETE_DIR_NAME in dirs:
                dirs.remove(DELETE_DIR_NAME)
            if PROCESSED_DIR_NAME in dirs:
                dirs.remove(PROCESSED_DIR_NAME)

            for filename in files:
                if filename.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".heic", ".heif")
                ):
                    filepath = os.path.join(root, filename)

                    # Submit task immediately
                    future = executor.submit(
                        process_single_file,
                        filepath,
                        model,
                        face_cascade_path,
                        delete_dir,
                        restore_map,
                        map_lock,
                    )
                    futures.add(future)

                    # Manage completed futures to keep memory usage low and update progress
                    # Check for completed tasks without blocking
                    done_futures = {f for f in futures if f.done()}
                    for f in done_futures:
                        processed_count += 1
                        if processed_count % 10 == 0:
                            print(f"Processed: {processed_count} files...", end="\r")
                        futures.remove(f)

        # Wait for remaining tasks
        for future in as_completed(futures):
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed: {processed_count} files...", end="\r")

    print("")  # Newline after progress
    save_restore_map(input_dir, restore_map)
    logging.info(
        f"--- DONE. Processed {processed_count} files. Log saved to {log_file_path} ---"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter photos based on subjects (Person/Pet)."
    )
    parser.add_argument("input_dir", help="Directory containing photos to scan")
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore mode: Move remaining files from 'to_delete' back to original locations",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker threads (default: 4)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist.")
        exit(1)

    process_photos(args.input_dir, args.restore, args.workers)
