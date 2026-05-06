import cv2


def clean_image(image):

    # mild enhancement
    cleaned = cv2.detailEnhance(
        image,
        sigma_s=5,
        sigma_r=0.05
    )

    return cleaned