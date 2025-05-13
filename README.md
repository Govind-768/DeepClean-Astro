# 🌌 DeepClean Astro: Autoencoder for Astronomical Image Denoising

<p align="center">
  <img src="https://img.shields.io/badge/Built%20With-Keras-blue?logo=keras" />
  <img src="https://img.shields.io/badge/Colab-Compatible-yellow?logo=googlecolab" />
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" />
  <a href="https://colab.research.google.com/github/your-username/image-denoising-astro/blob/main/denoising_autoencoder.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

> **DeepClean Astro** is a convolutional autoencoder built to clean outliers and noise from **astronomical grayscale images** — including artifacts from cosmic rays, background interference, or imaging hardware. This tool enhances visibility and clarity for deep-sky image analysis.

---

## ✨ Features

- 🧠 **Convolutional Autoencoder** trained on Noisy vs. Clean images
- 🧪 Evaluates denoising using **PSNR** and **SSIM**
- 📤 Upload & test custom images directly in **Colab**
- 🖼️ Visual comparison between **input** and **denoised output**
- 💾 Auto model saving/loading to avoid retraining
- 📦 Accepts `.zip` datasets with `Noisy/` and `Clean/` folders

---

## 🚀 Quick Start in Google Colab

1. Click below to launch in Colab  
   📎 [Open in Colab](https://colab.research.google.com/github/your-username/image-denoising-astro/blob/main/denoising_autoencoder.ipynb)

2. Upload a `dataset.zip` file containing:

└── dataset.zip
├── Noisy/
└── Clean/


3. Let the model train or load from `image_denoiser_model.h5`.

4. Upload your own astronomical `.jpg/.png` image to denoise it!

---

## 🖼️ Example Output

| Noisy Input | Denoised Output |
|-------------|-----------------|
|(![WhatsApp Image 2025-05-13 at 9 37 25 AM](https://github.com/user-attachments/assets/4eebee67-5044-45ff-aa6a-3861c119e295)) | ![WhatsApp Image 2025-05-13 at 9 37 24 AM](https://github.com/user-attachments/assets/9c31d6f4-bde8-48d1-b7bf-98fe77fc3af6)|

> *All images are grayscale and resized to 128×128.*

---

## 📊 Evaluation

- 🔍 **PSNR (Peak Signal-to-Noise Ratio)**: `~28.5 dB`
- 🧮 **SSIM (Structural Similarity Index)**: `~0.92`

---

## 📁 Folder Structure

📂 image-denoising-astro
│

├── denoising_autoencoder.ipynb # Main Colab-compatible notebook

├── image_denoiser_model.h5 # Saved model after training

├── assets/

│ ├── noisy_sample.png # Example noisy image

│ └── denoised_sample.png # Corresponding clean output

├── dataset.zip # (Optional) zipped dataset

└── README.md # This file


- Images should ideally be grayscale and related to astronomical content.

---

## 🧠 Model Architecture

- Encoder: 2 Convolutional layers + MaxPooling  
- Decoder: 2 UpSampling layers + 2 Convolutional layers  
- Loss: `MeanSquaredError`  
- Optimizer: `Adam`

---

## 🔘 Try It Live!

| Feature | Link |
|--------|------|
| 🧪 Run in Colab | [Open Notebook](https://colab.research.google.com/drive/1Rk6mJDmfAHJYowp8zS7isJGravmoOjPG?usp=sharing) |
| 📁 Sample Dataset | [Download](https://github.com/your-username/image-denoising-astro/releases/latest) |
| 📸 Upload Image & Denoise | Use the upload widget in the notebook |

---

## 📬 Feedback or Contributions

Have suggestions? Found a bug?  
Open an [issue](https://github.com/your-username/image-denoising-astro/issues) or submit a [pull request](https://github.com/Govind-768/DeepClean-Astro/compare)!

---

## 📜 License

This project is under the [MIT License](LICENSE).

---

> © 2025 - DeepClean Astro by [Govind Singh Rajput]
