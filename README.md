# Seam Carving Application

## About the Project

This project is a **GUI-based Seam Carving tool** implemented in Python. It uses **content-aware image resizing** by removing low-energy seams from an image while preserving important content. The application integrates **depth maps** and **saliency maps** to improve seam selection quality.

### Features

- GUI built with **Tkinter**
- Load images, depth maps, and saliency maps from a folder
- Seam carving with column or row removal
- Energy map calculation combining:
  - Image gradients (Sobel)
  - Laplacian edges
  - Depth map weighting
  - Saliency map weighting
- Real-time progress display with:
  - Progress bar
  - Seam visualization
  - Energy map visualization
- Interactive zoom and pan for image previews
- Save output image with custom scaling

---

## How It Works

The application computes an **energy map** using a combination of:
- Image gradients
- Depth map (object proximity importance)
- Saliency map (visual attention regions)

Low-energy seams (least important pixels) are removed iteratively along the chosen axis (rows or columns), resizing the image without distorting significant content.

---

## Installation

Clone the repository and install dependencies:

```
git clone https://github.com/your-username/seam-carving-app.git
cd seam-carving-app
pip install -r requirements.txt
```

---

## Usage

1. Run the application:

```
python SeamCarvingApp.py
```

2. In the GUI:
   - Click **Load Images from Folder** and select a folder containing:
     - Original image
     - Depth map (filename contains `_DMap`)
     - Saliency map (filename contains `_SMap`)
   - Choose seam removal **axis** (Row/Column)
   - Enter **scale** (e.g., 0.8 to remove 20% width/height)
   - Click **Run Seam Carving**
   - Save the output when prompted

---


## GUI Demo

- Load image and maps
- Visualize energy map and seams in real time
- Observe progress and elapsed time during carving
- Save the resized image



