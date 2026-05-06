# DeepClean-Astro

## Overview

DeepClean-Astro is an image preprocessing and enhancement project focused on improving image quality using computer vision techniques.

The project performs image preprocessing, enhancement, denoising, and comparison generation using OpenCV-based workflows. It was designed to understand how raw image data is processed before further analysis in computer vision systems.

---

## Features

- Image preprocessing pipeline
- Image enhancement and denoising
- Batch image processing
- Automatic comparison generation
- Modular project structure
- Output image saving
- Computer vision workflow implementation

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib

---

## Project Structure

```text
DeepClean-Astro/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── cleaned_image.png
│   ├── noisy_image.png
│   └── comparison_output.png
│
├── src/
│   ├── load_image.py
│   ├── preprocess.py
│   ├── denoise.py
│   ├── evaluate.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Workflow

1. Load raw images
2. Preprocess and resize images
3. Apply enhancement and denoising
4. Generate processed outputs
5. Save side-by-side comparisons

---

## Sample Output

### Image Comparison

![Comparison Output](outputs/comparison_output.png)

---

## Learnings

This project helped in understanding:

- Image preprocessing workflows
- Computer vision fundamentals
- Image enhancement techniques
- Noise reduction concepts
- Structured Python project organization

---

## Future Improvements

- Add deep learning based super-resolution
- Add GUI for image uploads
- Build API for image enhancement
- Add real-time processing support

---

## Author

Govind Singh