# **Adaptive Polynomial Image Reconstruction and Edge Detection**

This project implements an advanced image processing pipeline focused on adaptive polynomial approximation for efficient image reconstruction and subsequent edge detection leveraging the approximation errors. The methodology explores representing image information using polynomial coefficients over dynamically segmented regions, offering potential for data compression and resolution enhancement.

## **Table of Contents**

1. [Project Overview](#bookmark=id.pn1aa7a5j7qh)  
2. [Features](#bookmark=id.z63mvzby885k)  
3. [Project Structure](#bookmark=id.4xz03251y0y8)  
4. [Prerequisites](#bookmark=id.nyx4yliubrvh)  
5. [Setup and Installation](#bookmark=id.a5d3f6wwe10f)  
6. [Usage](#bookmark=id.69nj18a72oho)  
   * [Running Experiments](#bookmark=id.tsy7592zw7ac)  
   * [Configuration](#bookmark=id.w8v09kdlmt0k)  
   * [Exporting Results for Overleaf](#bookmark=id.b0e1vbfowrd3)  
   * [Computational Performance Analysis](#bookmark=id.2uqaykss3ki7)  
7. [Results](#bookmark=id.7x4i5d8nhdfk)  
8. [Troubleshooting](#bookmark=id.alml5cbeadj3)  
9. [License](#bookmark=id.s6idcd4ckiut)  
10. [Contributing](#bookmark=id.8h2hw5syv29j)

## **1\. Project Overview**

This repository provides a robust framework for performing adaptive image reconstruction using bivariate polynomial approximations. The image is recursively subdivided into segments based on a defined error criterion. For each terminal segment, polynomial coefficients are computed, effectively compressing the image information. These coefficients can then be used to reconstruct the image, potentially at higher resolutions. Furthermore, the error maps generated during this reconstruction process are utilized for various edge detection strategies.

## **2\. Features**

* **Adaptive Image Reconstruction:** Dynamic segmentation of images based on local approximation errors, using a quadtree-like subdivision strategy.  
* **Polynomial Approximation:** Utilizes various interpolation node methods (full-mesh, Leja, Fekete, Padua) and polynomial bases (Monomial, Chebyshev) for approximating image segments.  
* **Parallel Processing:** Leverages Python's multiprocessing module to accelerate segment approximation across multiple CPU cores.  
* **Resolution Enhancement:** Reconstructs images at higher resolutions from their compact polynomial coefficient representation, without requiring additional high-resolution source data.  
* **Error-Based Edge Detection:** Detects image edges by analyzing the error maps resulting from the polynomial approximation, offering various combination strategies (e.g., maximum error, weighted sum of errors) and thresholding methods.  
* **Comprehensive Visualization:** Generates a suite of high-quality plots for detailed analysis, including:  
  * Original and Reconstructed Images  
  * Actual Reconstruction Error Heatmaps  
  * Reconstruction with Segmentation Boundaries  
  * Segment Error Heatmaps  
  * Statistical plots: Error Distribution, Depth vs. Segment Count, Segment Size Distribution.  
  * Comparison Plots for different Edge Detection Strategies and their raw error maps.  
* **Computational Performance Analysis:** Includes a dedicated script to measure and compare the execution time of different polynomial approximation node generation methods.  
* **Overleaf Export Utility:** A utility script to automatically create copies of generated result plots with shortened, Overleaf-compatible filenames.

## **3\. Project Structure**

.  
├── config/                     \# Configuration files for reconstruction and edge detection parameters  
│   ├── \_\_init\_\_.py             \# Makes config a Python package  
│   ├── config\_detector.py      \# Edge detection specific configurations  
│   └── config\_reconstructor.py \# Image reconstruction specific configurations  
├── docs/                       \# (Placeholder for additional documentation, if any)  
├── images/                     \# Input images (e.g., Shepp\_Logan\_phantom.png, spiral.png)  
├── overleaf\_exports/           \# \*\*(Generated \- untracked by Git)\*\* Exported plots/data with shortened filenames for Overleaf upload  
├── poly\_approx/                \# Core mathematical and image processing algorithms for polynomial approximation  
│   ├── \_\_init\_\_.py             \# Makes poly\_approx a Python package  
│   ├── admissible\_meshes.py    \# Functions for generating admissible meshes (e.g., Chebyshev)  
│   ├── edge\_processing.py      \# Functions for combining error maps and applying thresholds  
│   ├── image\_poly\_approximation.py \# Main logic for polynomial approximation of image segments  
│   ├── image\_reconstruction\_metrics.py \# Functions for calculating error metrics and normalizing images  
│   ├── interpolation\_nodes.py  \# Functions for generating different types of interpolation nodes  
│   ├── poly\_projector.py       \# Functions for evaluating polynomials from coefficients  
│   └── polynomial\_bases.py     \# Definitions for different polynomial bases and multi-indices  
├── results/                    \# \*\*(Generated \- untracked by Git)\*\* All raw output results (plots, data, etc.)  
│   ├── admissible\_mesh\_tests/      \# Test results for admissible mesh generation  
│   ├── computational\_performance\_results/ \# Performance analysis outputs (.json)  
│   ├── edge\_detection\_results/     \# Edge detection specific plots  
│   ├── image\_reconstruction\_results/ \# Image reconstruction plots and segment data (.pkl)  
│   ├── interpolation\_node\_tests/   \# Test results for interpolation node generation  
│   └── poly\_projector\_tests/       \# Test results for polynomial projection  
├── scripts/                    \# Helper scripts and utility tools  
│   ├── canny\_edge\_detect.py    \# (Optional) For Canny edge detection comparison  
│   ├── export\_for\_overleaf.py  \# Script to shorten filenames for Overleaf export  
│   ├── test\_admissible\_mesh.py \# Test script for admissible mesh generation  
│   ├── test\_interpolation\_nodes.py \# Test script for interpolation node generation  
│   └── test\_poly\_projector.py  \# Test script for polynomial projection  
├── utils/                      \# Utility modules that integrate core functionalities  
│   ├── \_\_init\_\_.py             \# Makes utils a Python package  
│   ├── computational\_performance\_analyzer.py \# Analyzes performance of approximation methods  
│   ├── edge\_detector.py        \# Implements the edge detection pipeline  
│   └── image\_reconstructor.py  \# Implements the adaptive image reconstruction pipeline  
├── venv/                       \# \*\*(Generated \- untracked by Git)\*\* Python virtual environment  
├── .gitignore                  \# Specifies intentionally untracked files and directories  
├── LICENSE                     \# Project license (e.g., MIT, Apache 2.0) \- \*\*Please fill in details\*\*  
├── README.md                   \# Project overview and instructions  
├── requirements.txt            \# Lists Python package dependencies  
└── run\_experiments.py          \# Main orchestrator script to run reconstruction and edge detection

## **4\. Prerequisites**

Before you begin, ensure you have the following installed:

* **Python 3.12.4** (or compatible versions 3.8+)

## **5\. Setup and Installation**

To set up and run this project, follow these steps:

1. Clone the Repository:  
   Navigate to the directory where you want to store the project and clone the repository:  
   git clone https://github.com/your-username/your-repo-name.git \# Replace with your actual repo URL  
   cd your-repo-name

2. Create a Python Virtual Environment (Recommended):  
   A virtual environment helps manage project dependencies without interfering with your system's global Python packages.  
   python \-m venv venv

3. **Activate the Virtual Environment:**  
   * **On Windows:**  
     .\\venv\\Scripts\\activate

   * **On macOS/Linux:**  
     source venv/bin/activate

4. Install Dependencies:  
   Install all required Python packages listed in requirements.txt:  
   pip install \-r requirements.txt

   *If requirements.txt is missing or outdated, you can generate it from your current environment using:*  
   pip freeze \> requirements.txt

## **6\. Usage**

### **6.1. Running Experiments**

The central script for orchestrating image reconstruction and edge detection experiments is run\_experiments.py. It automates the process for a set of predefined images, checking for existing reconstruction data to avoid redundant computations.

To execute the full experimental pipeline:

python run\_experiments.py

This script will process each image, perform adaptive reconstruction (if its data (.pkl file) is not already generated), and then apply various edge detection strategies, saving all generated plots and data files to the results/ directory.

### **6.2. Configuration**

All configurable parameters for the image reconstruction and edge detection processes are located in the config/ directory. You can adjust these files to customize the behavior of the algorithms:

* **config/config\_reconstructor.py**:  
  * Controls parameters for the adaptive segmentation and polynomial approximation, such as polynomial degree (POLY\_DEGREE), node generation method (NODES\_METHOD), error criteria (ERROR\_THRESHOLD, ERROR\_MEASURE\_TYPE), recursion limits (MAX\_DEPTH, MIN\_SEGMENT\_SIZE), and parallel processing (NUM\_PROCESSES). Also defines flags for saving reconstruction-specific plots and data.  
* **config/config\_detector.py**:  
  * Manages parameters for edge detection, including the strategy for combining error maps (ERROR\_COMBINATION\_STRATEGY, ERROR\_COMBINATION\_WEIGHTS), thresholding type (EDGE\_THRESHOLD\_TYPE, FIXED\_EDGE\_THRESHOLD), and which strategies to compare in plots (EDGE\_STRATEGIES\_TO\_COMPARE).

Note on Image-Specific Thresholds:  
The run\_experiments.py script overrides the FIXED\_EDGE\_THRESHOLD from config\_detector.py with specific values for each image during the edge detection phase (e.g., for Shepp\_Logan\_phantom, spiral). You can modify these values directly within the images\_to\_process list in run\_experiments.py to fine-tune edge detection thresholds per image.

### **6.3. Exporting Results for Overleaf**

To prepare your generated plots for inclusion in LaTeX documents (e.g., on Overleaf), which often have filename length restrictions, use the export\_for\_overleaf.py script:

python scripts/export\_for\_overleaf.py

This script will create a new top-level directory named overleaf\_exports/ in your project root. Inside, you'll find copies of your results/ files with systematically shortened, more manageable filenames, organized into the same subdirectories.

**Example of including a shortened figure in your LaTeX document:**

\\includegraphics\[width=0.45\\textwidth\]{overleaf\_exports/edge\_detection\_results/shepp\_logan\_phantom/shepp\_logan\_phantom\_comp\_d5e0001sWS.pdf}

*(Note: The exact shortened filename will depend on the original plot type and parameters.)*

### **6.4. Computational Performance Analysis**

To run an analysis of the computational performance of different polynomial approximation node generation methods:

python utils/computational\_performance\_analyzer.py

This script will print a summary table to the console and save detailed performance results to a JSON file in results/computational\_performance\_results/.

## **7\. Results**

All generated plots (PDFs, PNGs) and data files (.pkl, .json) are saved into the results/ directory, organized by image and experiment type. For Overleaf-compatible versions, refer to the overleaf\_exports/ directory after running the export script.

Consider adding a screenshot of some example output plots here to give a quick visual overview of the project's capabilities.

## **8\. Troubleshooting**

* **ImportError: No module named '...'**:  
  * Ensure your virtual environment is activated (source venv/bin/activate or .\\venv\\Scripts\\activate).  
  * Confirm all dependencies are installed (pip install \-r requirements.txt).  
  * Verify your project structure matches the one described above, especially for imports from config/, utils/, and poly\_approx/.  
* **"Error loading image: FileNotFoundError"**:  
  * Make sure the images specified in config\_reconstructor.py and run\_experiments.py actually exist in the images/ directory.  
* **"Reconstruction data NOT found..." when it should be there**:  
  * Check if the results/image\_reconstruction\_results/ directory exists and contains the .pkl files. Filenames are sensitive to reconstruction parameters; if you changed config\_reconstructor.py since the last run, the old .pkl might not match the new expected filename.

## **9\. License**

This project is licensed under the **MIT License**. See the LICENSE file for details.

## **10\. Contributing**

We welcome contributions\! If you'd like to contribute, please refer to the following guidelines:

* Fork the repository.  
* Create a new branch for your feature or bug fix.  
* Ensure your code adheres to the project's style guidelines.  
* Write clear commit messages.  
* Submit a pull request.