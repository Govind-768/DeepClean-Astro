import matplotlib.pyplot as plt
import cv2
import os


def save_comparison(
    original,
    cleaned,
    image_name
):

    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    cleaned_rgb = cv2.cvtColor(
        cleaned,
        cv2.COLOR_BGR2RGB
    )

    base_name = os.path.splitext(
        image_name
    )[0]

    cv2.imwrite(
        f"outputs/{base_name}_cleaned.png",
        cleaned
    )

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cleaned_rgb)
    plt.title("Processed")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        f"outputs/{base_name}_comparison.png"
    )

    plt.close()