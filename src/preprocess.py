import cv2


def preprocess_image(image):

    # upscale image
    image = cv2.resize(
        image,
        (1024, 1024),
        interpolation=cv2.INTER_CUBIC
    )

    # improve contrast
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))

    enhanced_image = cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )

    return enhanced_image