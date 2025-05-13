import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from google.colab import files
from zipfile import ZipFile
import ipywidgets as widgets
from IPython.display import display, clear_output

print("Upload your dataset.zip containing Clean and Noisy folders")
uploaded = files.upload()

for file in uploaded:
    if file.endswith(".zip"):
        with ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall("content")

def get_folder(name):
    for root, dirs, _ in os.walk("content"):
        for dir_name in dirs:
            if dir_name.lower() == name.lower():
                return os.path.join(root, dir_name)
    raise FileNotFoundError(f"{name} folder not found")

noisy_dir = get_folder("Noisy")
clean_dir = get_folder("Clean")

def read_images(folder, size=(128, 128)):
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_path in glob(os.path.join(folder, ext)):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, size)
            images.append(np.expand_dims(img / 255.0, axis=-1))
    return np.array(images)

x = read_images(noisy_dir)
y = read_images(clean_dir)

min_len = min(len(x), len(y))
x, y = x[:min_len], y[:min_len]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

def autoencoder():
    i = Input(shape=(128, 128, 1))
    e = Conv2D(32, 3, activation='relu', padding='same')(i)
    e = MaxPooling2D(2, padding='same')(e)
    e = Conv2D(16, 3, activation='relu', padding='same')(e)
    e = MaxPooling2D(2, padding='same')(e)

    d = UpSampling2D(2)(e)
    d = Conv2D(16, 3, activation='relu', padding='same')(d)
    d = UpSampling2D(2)(d)
    d = Conv2D(32, 3, activation='relu', padding='same')(d)
    d = Conv2D(1, 3, activation='sigmoid', padding='same')(d)

    model = Model(i, d)
    model.compile(optimizer=Adam(), loss=MeanSquaredError())
    return model

model_path = "image_denoiser_model.h5"

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = autoencoder()
    model.fit(X_train, Y_train, epochs=10, batch_size=4, validation_split=0.1)
    model.save(model_path)

psnr_values, ssim_values = [], []
data_range = 1.0

for i in range(len(X_test)):
    clean = Y_test[i].squeeze()
    pred = model.predict(np.expand_dims(X_test[i], axis=0))[0].squeeze()

    psnr = peak_signal_noise_ratio(clean, pred, data_range=data_range)
    ssim = structural_similarity(clean, pred, data_range=data_range)
    psnr_values.append(psnr)
    ssim_values.append(ssim)

print(f"Average PSNR: {np.mean(psnr_values):.4f}")
print(f"Average SSIM: {np.mean(ssim_values):.4f}")

upload_button = widgets.FileUpload(accept=".jpg,.jpeg,.png", multiple=False)

def on_image_upload(change):
    clear_output(wait=True)
    display(upload_button)

    img_data = list(change['new'].values())[0]['content']
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 128)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=(0, -1))

    pred = model.predict(img_resized)[0, ..., 0]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title("Uploaded Image")
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(pred, cmap='gray')
    plt.show()

upload_button.observe(on_image_upload, names='value')
display(upload_button)

print("\nUse the UI to upload and test a custom image.")
