import os

from src.load_image import load_image
from src.preprocess import preprocess_image
from src.denoise import clean_image
from src.evaluate import save_comparison


def main():

    input_folder = "data/raw"

    image_files = os.listdir(input_folder)

    print(f"Found {len(image_files)} images")

    for image_name in image_files:

        image_path = os.path.join(
            input_folder,
            image_name
        )

        print(f"\nProcessing: {image_name}")

        image = load_image(image_path)

        processed_image = preprocess_image(image)

        cleaned_image = clean_image(
            processed_image
        )

        save_comparison(
            image,
            cleaned_image,
            image_name
        )

    print("\nAll images processed successfully.")


if __name__ == "__main__":
    main()