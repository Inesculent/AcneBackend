import torch
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from retinaface import RetinaFace
import os

from CFG import CFG

def preprocess_image(image):
    if isinstance(image, tf.Tensor) and not tf.executing_eagerly():
        # Convert symbolic tensor to a NumPy array in graph mode
        image = image.numpy()
    elif isinstance(image, tf.Tensor):
        # Ensure eager tensor compatibility
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
    elif isinstance(image, np.ndarray):
        # Ensure correct type for NumPy array
        image = image.astype(np.uint8)
    return image

def face_area(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Currently not working owing to runtime environment conflicts with uvicorn.
    # WILL try to fix soon.
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(100,100)
    )

    for (x, y, w, h) in face:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 10))
    plt.imshow(img_rgb)
    plt.axis('off')

# Currently has runtime environment issues.
def find_face(image):

    try:
        img = preprocess_image(image)
        print("Here")
        faces = RetinaFace.extract_faces(img)
    except Exception as e:
        print(f"Error in RetinaFace.extract_faces: {e}")
        return None


    for face in faces:
        if face is not None:
            height, width = face.shape[:2]

            # Convert BGR to RGB (if using OpenCV to load the image)
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Display the face in correct colors
            plt.figure(figsize=(4, 4))
            plt.imshow(face_rgb)
            plt.axis("off")
            plt.show()

            return face_rgb  # Return the corrected face

    return None

# Function to plot the predictions
def visualize_prediction(model, image_path, transform, device, threshold=0.5):
    """
    Visualize original and transformed images, masks, and overlays in two rows.

    Args:
        model (torch.nn.Module): Trained segmentation model.
        image_path (str): Path to the input image.
        mask_path (str): Path to the corresponding ground truth mask.
        transform (albumentations.Compose): Transformation to apply to the image and mask.
        device (torch.device): Device (CPU or CUDA).
        threshold (float): Threshold for binary segmentation (default is 0.5).

    Returns:
        None
    """
    # Load the original image and mask
    img = np.array(Image.open(image_path).convert("RGB"))

    face = find_face(img)
    #face = None

    if face is not None:
        # Apply transformations to get the "new" versions
        transformed = transform(image=face)
    else:
        transformed = transform(image=img)

    transformed_img = transformed['image']



    img_tensor = torch.tensor(transformed_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Run the model to predict the mask
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor)  # Forward pass
        if pred.shape[1] == 1:  # Binary segmentation
            pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
            pred_mask = (pred_mask > threshold).astype("uint8")
        else:  # Multiclass segmentation
            pred_mask = torch.argmax(pred.squeeze(0), dim=0).cpu().numpy()


    num_pixels = np.sum(pred_mask)
    width, height = pred_mask.shape[:2]
    size = width * height


    coverage = round((num_pixels / size) * 100, 2)

    num_objects, labeled_mask = cv2.connectedComponents(pred_mask, connectivity=8)

    print("Number of acne cells:", num_objects)


    # Debugging
    # print("Predicted Mask Shape:", pred_mask.shape)
    # print("Unique Values in Predicted Mask:", np.unique(pred_mask))
    # print("Number of pixels in Predicted Mask:", num_pixels)
    # print("Total pixels:", size)
    # print("Percentage of acne coverage:", (num_pixels / size) * 100)


    # Create overlays
    overlay_transformed = transformed_img.copy()


    # Overlay predicted mask in red
    overlay_transformed[pred_mask > 0] = [255, 0, 0]

    # Visualize results
    plt.figure(figsize=(20, 12))


    # All the different "subplots" to showcase before and after
    plt.subplot(1, 3, 1)
    plt.imshow(transformed_img.astype("uint8"))
    plt.title("Transformed Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask.astype("uint8"))
    plt.title("Mask")
    plt.axis("off")


    plt.subplot(1, 3, 3)
    plt.imshow(overlay_transformed.astype("uint8"))
    plt.title("Overlay (Prediction)")
    plt.axis("off")

    plt.tight_layout()

    im_name = os.path.basename(image_path)

    file_path = "C:\\Users\\parlo\\PycharmProjects\\AcneDetector\\generated_images\\" + im_name

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    plt.savefig(file_path , dpi=300, bbox_inches="tight")

    print(file_path)
    plt.show()



    return file_path, coverage, num_objects

# To run the segmentation
def run_segment(path, model):

    # Recreate the model

    file_path, coverage, num_objects = visualize_prediction(model, path, CFG.data_transforms["valid"], CFG.device)

    return file_path, coverage, num_objects


