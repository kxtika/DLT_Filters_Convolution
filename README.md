# Computer Vision Coursework

This repository contains Python implementations developed as part of a Computer Vision coursework. It includes two main components:

1. **DLT Camera Calibration** (`DLT/` folder)  
2. **Filtering, Convolution, and Edge Detection** (`Filters_convolution/` folder)

## 📁 Repository Structure

```
DLT_Filters_Convolution/
├── DLT/
│   ├── main.py                 # Direct Linear Transform, decomposition and reprojection code
│   ├── points.py               # Script to collect 2D coordinates of the 3D cube          
│   └── checkerboard_cube/      # Images for calibration and identifying the origin
│         ├── cube0.jpg         # Image used for calibration
│         ├── cube_origin.png   # Image used for origin identification
│  
├── Filters_convolution/
│   ├── custom_filters.py       # Custom filters and edge detection
│   ├── opencv_filters.py       # OpenCV-based implementation
│   └── image.jpg               # Sample input image
├── requireemnts.txt
└── README.md
```

## 📌 Features

### ✅ DLT Camera Calibration (`DLT/`)
- Manual collection of 2D points from image using OpenCV  
- Projection matrix estimation using Direct Linear Transform (DLT)  
- Intrinsic and extrinsic parameter decomposition  
- Reprojection error computation and visualisation  

### ✅ Image Filtering and Edge Detection (`Filters_convolution/`)
- **Custom implementations** of:
  - Gaussian, Mean, Median, and Gabor filters  
  - 2D convolution with custom kernels  
  - Gradient-based edge detection with hysteresis  
- **OpenCV equivalents** for performance and quality comparison  
- Command-line interface with timing outputs  

## 🚀 Running the Scripts

Install dependencies:
```bash
pip install -r requirements.txt
```

Run a custom filter:
```bash
python Filters_convolution/custom_filters.py Filters_convolution/image.jpg 5 gaussian 1.0 50 100
```

Run with OpenCV:
```bash
python Filters_convolution/opencv_filters.py Filters_convolution/image.jpg 5 gaussian 1.0 50 100
```

## Author

**Kateryna Miniailo**  
Liverpool Hope University  
2025
