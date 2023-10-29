import torch
import cv2
import numpy as np
from model import SandNet

"""
Predicts the soil type of an input image.

Parameters:
    image_path (str): The path to the input image.

Returns:
    str: The predicted soil type.
"""
def predict_soil_type(image_path):
    # Load the trained model
    model = SandNet()
    model.load_state_dict(torch.load('path_to_trained_model.pth'))
    model.eval()

    # Preprocess the input image
    input_image = load_image(image_path)

    # Pass the image through the model
    with torch.no_grad():
        output = model(input_image)
        predicted_class = torch.argmax(output).item()

    # Map the predicted class index to the corresponding soil type
    soil_types = ['Clay', 'Loam', 'Sand', 'Sandy Loam', 'Silt']
    predicted_soil_type = soil_types[predicted_class]

    return predicted_soil_type

"""
Load and preprocess an image from the given file path.

Parameters:
    file_path (str): The path to the image file.

Returns:
    torch.Tensor: The preprocessed image tensor.
"""
def load_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).float()

    return img
