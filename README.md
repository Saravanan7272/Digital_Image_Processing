# Digital Image Processing

This repository contains a single Jupyter notebook, `IPA_ASS.ipynb`, which implements an image-processing pipeline covering image acquisition, noise simulation, denoising/filters, enhancement, segmentation, feature extraction, and a final pipeline visualization and reflection.

This README documents the notebook, its outputs, how to run it (locally or in Google Colab), dependencies, and the set of files and folders the notebook produces when executed successfully.

## Notebook overview

- File: `IPA_ASS.ipynb`
- Purpose: Demonstrates a full image-processing assignment (Tasks 1–7). The notebook:
  - Loads an input image (`image.jpeg`).
  - Computes basic statistics and color/grayscale histograms.
  - Adds three types of synthetic noise (Gaussian, Salt & Pepper, Speckle) at multiple strengths and reports MSE, PSNR, and SSIM.
  - Saves noisy examples to `noisy_outputs/`.
  - Applies simple denoising filters (median and Gaussian), compares performance, and saves to `filter_outputs/`.
  - Performs segmentation (Otsu thresholding and K-Means color clustering), creates masks and overlays, and saves to `segmentation_outputs/`.
  - Extracts region features (area, centroid, bbox, color histograms), annotates and saves feature images to `features_outputs/`.
  - Assembles a pipeline visualization image and a short reflection text in `pipeline_outputs/`.

## Expected output folders (created by the notebook)

- `noisy_outputs/` — PNG images with synthetic noise applied. Filenames are like `Gaussian_15.png`, `SaltPepper_0p03.png`, `Speckle_0p05.png`.
- `filter_outputs/` — Filtered outputs (median and Gaussian) saved from noisy inputs.
- `segmentation_outputs/` — Segmentation masks and overlay visualizations (e.g., `otsu_mask.png`, `kmeans_mask.png`, `otsu_overlay.png`, `kmeans_overlay.png`, `canny_edges.png`).
- `features_outputs/` — Annotated images showing detected contours, centroids and bounding boxes plus saved color-histogram plots (`otsu_features.png`, `kmeans_features.png`).
- `pipeline_outputs/` — Full pipeline overview image (`pipeline_overview.png`) and a small textual reflection `pipeline_summary.txt`.

> Note: The notebook will create these folders automatically if they do not already exist.

## Required files

- `image.jpeg` — The input image required by the notebook. Place this file in the repository root (same folder as `IPA_ASS.ipynb`) before running the notebook locally. The notebook assumes the path `image.jpeg` (or `/content/image.jpeg` in Colab).

## Dependencies

The notebook uses OpenCV, NumPy, Matplotlib, pandas and scikit-image. A minimal requirements list is below.

Recommended (pip):

```bash
python -m pip install --upgrade pip
pip install numpy opencv-python matplotlib pandas scikit-image
```

If you run inside Google Colab, many of these packages are pre-installed; still, you may want to run:

```python
!pip install --upgrade scikit-image opencv-python-headless
```

Notes:
- On Windows with WSL, install into the Python environment you will use to run Jupyter/VSCode. The notebook was developed with standard CPU libraries (no GPU-specific packages).
- Use `opencv-python` for local runs. On some setups `opencv-python-headless` is preferred for headless servers.

## How to run

1. Local (VS Code / Jupyter):
   - Ensure `image.jpeg` is in the same folder as `IPA_ASS.ipynb`.
   - Create a virtual environment (optional):

```bash
python -m venv .venv
source .venv/bin/activate   # WSL / macOS / Linux
# .venv\Scripts\activate   # Windows (cmd/powershell)
pip install -r requirements.txt   # or install packages listed above
```

   - Open `IPA_ASS.ipynb` in VS Code or Jupyter Notebook and run the cells top-to-bottom. The notebook is structured in logical sections (acquisition & statistics, noise simulation, filtering, enhancement, segmentation, feature extraction, pipeline visualization).

2. Google Colab:
   - Click the Colab badge at the top of the notebook (already present in the notebook). If running in Colab, upload `image.jpeg` to `/content/` or modify `IMAGE_PATH` to point to the uploaded file.
   - Run the notebook cells sequentially.

## Important runtime notes and assumptions

- The notebook expects to find `image.jpeg` in the working directory. If your file has a different name or path, update the `IMAGE_PATH` variable at the top of the notebook.
- The code saves intermediate outputs to the folders listed above; verify you have write permissions in the repository directory.
- The notebook calculates MSE, PSNR and SSIM (grayscale-based SSIM used for comparisons). Numeric values are printed and some are stored into pandas DataFrames for display.
- Some function implementations assume the input images are 8-bit (0–255). If you change data ranges, adapt scaling accordingly.

## What each notebook section does (brief)

- Image acquisition & statistics: loads the image and prints shape, data type, min/max/mean and per-channel stats. Plots RGB and grayscale histograms.
- Noise simulation: creates multiple noisy variants (Gaussian, Salt & Pepper, Speckle) at different strengths, computes MSE/PSNR/SSIM, and saves examples to `noisy_outputs/`.
- Filtering & denoising: applies median and Gaussian filters to noisy images, computes PSNR/SSIM improvements, displays comparisons, and saves results to `filter_outputs/`.
- Enhancement: resizing, histogram equalization, and contrast stretching for visualization.
- Segmentation: Otsu thresholding (binary) and K-Means color clustering (k=4). Cleans masks with morphology, computes IoU/Dice between masks, and saves mask and overlay images to `segmentation_outputs/`.
- Feature extraction: extracts centroid, bounding box, area and color histograms for segmented regions, annotates these on the original image and saves images to `features_outputs/`.
- Pipeline assembly & reflection: loads intermediate outputs and assembles a multi-panel overview, saves it to `pipeline_outputs/`, and writes a short reflection to `pipeline_summary.txt`.

## Reproducing results and verifying a successful run

1. Ensure `image.jpeg` is in repository root.
2. Run all cells in `IPA_ASS.ipynb` from top to bottom. Look for printed summaries (dataframes of metrics) and the saved output folders.
3. Verify these files exist after the run:

```
noisy_outputs/
filter_outputs/
segmentation_outputs/
features_outputs/
pipeline_outputs/pipeline_overview.png
pipeline_outputs/pipeline_summary.txt
```

4. Open `pipeline_outputs/pipeline_overview.png` to inspect the assembled visualization of the full pipeline.

## Troubleshooting

- If the notebook fails to find images: confirm `image.jpeg` is in the same folder and not corrupted.
- If OpenCV cannot be imported: install `opencv-python`.
- If a particular output image is missing, review the cell outputs in the notebook for printed warnings (the notebook logs warnings when it cannot find expected files).

## Suggested small improvements (next steps)

- Add a small `requirements.txt` (or `environment.yml`) to pin versions used during development.
- Add a minimal unit test script that checks the notebook-generated folders and a couple of expected image files exist after a headless run.
- Convert parts of the notebook into modular Python scripts for automated runs (e.g., `make_noisy.py`, `apply_filters.py`, `segment.py`) for reproducible pipelines.

