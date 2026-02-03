# PhotoOrg: Intelligent Photo Filter 📸

A powerful, AI-driven photo filtering tool designed for NAS and large photo libraries. It automatically scans your photos and filters out images that don't contain specific subjects (People with visible faces, Cats, or Dogs), helping you keep your collection clean and meaningful.

## ✨ Features

*   **AI-Powered Detection**: Uses **YOLOv8** for object detection and **OpenCV** for face verification.
*   **Smart Filtering**:
    *   **Keep**: Photos with **Cats**, **Dogs**, or **People with visible faces**.
    *   **Filter**: Photos without these subjects, or people/portraits where no face is detected (e.g., back view, blurry).
*   **Non-Destructive**: "Kept" photos remain untouched in their original location. "Filtered" photos are moved to a `to_delete` folder for review.
*   **Restore Mode**: Easily restore files from `to_delete` back to their original nested paths if you decide to keep them.
*   **HEIC Support**: Native support for Apple's HEIC format.
*   **High Performance**: Multi-threaded scanning (configurable workers) for fast processing of thousands of photos.
*   **Safe**: Includes "Dry Run" logic via the review workflow and strict directory pruning to prevent infinite loops.

## 🚀 Installation

This project uses `uv` for fast Python package management.

1.  **Install `uv`** (if not installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the repository**:
    ```bash
    git clone https://github.com/jasoft/photoorg.git
    cd photoorg
    ```

3.  **Install dependencies**:
    ```bash
    uv sync
    ```

    *Note: The script will automatically download the `yolov8n.pt` model weights (approx. 6MB) on the first run. You don't need to download it manually.*

## 📖 Usage

Run the script using `uv run`.

### 1. Basic Scan
Scan a directory (and all subdirectories) for photos.
```bash
uv run filter_photos.py /path/to/your/photos
```
*   **Result**: 
    *   Valid photos stay where they are.
    *   Invalid photos are moved to `/path/to/your/photos/to_delete`.
    *   A log file is created at `/path/to/your/photos/scan.log`.

### 2. High-Performance Scan
For large libraries or powerful machines (e.g., Mac Mini M4, Servers), increase the worker threads.
```bash
uv run filter_photos.py /path/to/your/photos --workers 10
```
*   **Recommendation**: Set `--workers` to 1.0x - 1.5x your CPU core count.

### 3. Restore Mode
If you review the `to_delete` folder and decide to **keep** some files (by leaving them there), run this command to put them back exactly where they came from.
```bash
uv run filter_photos.py /path/to/your/photos --restore
```
*   **Workflow**: 
    1. Run scan.
    2. Go to `to_delete` folder.
    3. **Delete** the photos you truly don't want.
    4. Run `--restore`. The remaining files in `to_delete` will be moved back to their original folders.

## ⚙️ Command Line Arguments

```text
usage: filter_photos.py [-h] [--restore] [--workers WORKERS] [input_dir]

Filter photos based on subjects (Person/Pet).

positional arguments:
  input_dir          Directory containing photos to scan

options:
  -h, --help         show this help message and exit
  --restore          Restore mode: Move remaining files from 'to_delete' back to original locations
  --workers WORKERS  Number of parallel worker threads (default: 4)
```

## 🛠️ Technical Details

*   **YOLOv8 Nano**: Uses the lightest YOLO model (`yolov8n.pt`) for a balance of speed and accuracy.
*   **Face Detection**: Uses OpenCV's Haar Cascade for confirming face visibility when a person is detected.
*   **Thread Safety**: Implements Thread-Local Storage for OpenCV classifiers to ensure stability during parallel processing.
*   **Format Support**: JPG, JPEG, PNG, HEIC, HEIF.

## 📝 License

MIT License
