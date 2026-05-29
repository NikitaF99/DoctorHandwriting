import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
import pandas as pd
import numpy as np
import os

def preprocess_data(df_train, df_test, df_val):
    print("Preprocessing data...")
    processed_train = [apply_preprocessing_to_row(row, '../dataset/Training/training_words') for _, row in df_train.iterrows()]
    processed_test = [apply_preprocessing_to_row(row, '../dataset/Testing/testing_words') for _, row in df_test.iterrows()]
    processed_val = [apply_preprocessing_to_row(row, '../dataset/Validation/validation_words') for _, row in df_val.iterrows()]

    for df, processed in zip([df_train, df_test, df_val],
                             [processed_train, processed_test, processed_val]):
        processed_df = pd.DataFrame(processed, columns=['preprocessed_image', 'processed_label'])
        df['preprocessed_image'] = processed_df['preprocessed_image']
        df['processed_label'] = processed_df['processed_label']

    all_labels = pd.concat([df_train['processed_label'], df_test['processed_label'], df_val['processed_label']])
    all_chars = sorted(list(set(''.join(all_labels.astype(str).tolist()))))
    char_to_int = {char: i for i, char in enumerate(all_chars)}
    int_to_char = {i: char for i, char in enumerate(all_chars)}
    max_label_length = max(all_labels.astype(str).apply(len))

    # Convert images to tensors and labels to CTC format
    for df in [df_train, df_test, df_val]:
        df['preprocessed_image_crnn'] = df['preprocessed_image'].apply(image_to_tensor)
        df['ctc_label'] = df['processed_label'].apply(lambda x: create_ctc_labels(x, char_to_int, max_label_length))

    print("Training DataFrame with CRNN preprocessed images and CTC labels:")
    print(df_train.head())

    print("\nValidation DataFrame with CRNN preprocessed images and CTC labels:")
    print(df_val.head())

    print("\nTesting DataFrame with CRNN preprocessed images and CTC labels:")
    print(df_test.head())

    return df_train, df_test, df_val


def preprocess_image(image_path, label):
    """Loads, resizes, converts to grayscale, and preprocesses an image."""

    img = cv2.imread(image_path)

    if img is None:
        print(f"Warning: Could not load image from {image_path}")
        return None, label 

    # Resize image
    img_resized = cv2.resize(img, (256, 128))

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresholded = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)


    # Normalize pixel values
    img_normalized = thresholded / 255.0 

    return img_normalized, label



def display_img(im_path):
  dpi = 80
  im_data = plt.imread(im_path)
  height, width = im_data.shape[:2]
  figsize = width / float(dpi), height / float(dpi)
  fig = plt.figure(figsize=figsize)
  ax = fig.add_axes([0, 0, 1, 1])
  ax.axis('off')
  ax.imshow(im_data, cmap='gray')
  plt.show()



def apply_preprocessing_to_row(row, base_image_dir):
    """Applies preprocessing to a single row of the DataFrame and returns a tuple."""
    image_path = os.path.join(base_image_dir, row['IMAGE'])
    label = row['MEDICINE_NAME'] 
    processed_image, processed_label = preprocess_image(image_path, label)
    return (processed_image, processed_label) # Return as a tuple



def image_to_tensor(image_array, target_height=32, max_width=256):
    """
    Preprocesses an image array to image tensor with consistent height and padded width.
    """
    
    # Convert NumPy array to PyTorch tensor
    img_tensor = torch.from_numpy(image_array).float()

    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0) 
    elif img_tensor.shape[-1] == 3:
        img_tensor = img_tensor.permute(2, 0, 1) 
        img_tensor = F.rgb_to_grayscale(img_tensor) 

    original_height = img_tensor.shape[1]
    original_width = img_tensor.shape[2]

    # Calculate new width 
    new_width = int(original_width * target_height / original_height)

    img_resized = F.resize(img_tensor, size=[target_height, new_width])

    # Pad or crop the width
    padding_width = max_width - new_width
    if padding_width > 0:
        # Pad the image on the right side
        img_padded = F.pad(img_resized, padding=[0, 0, padding_width, 0])
    else:
        # Crop the image if the new width exceeds max_width
        img_padded = img_resized[:, :, :max_width]


    return img_padded

# Create a character to integer mapping and vice versa
# Get all unique characters from the labels


def create_ctc_labels(label, char_to_int, max_label_length=None):
    """
    Creates CTC-friendly labels from a text label.
    Returns:
        A list of integers representing the label, padded if max_label_length is provided.
    """
    ctc_label = [char_to_int[char] for char in str(label) if char in char_to_int]
    if max_label_length:
        padding_value = len(char_to_int) # Use an integer outside the valid character range for padding
        ctc_label = ctc_label + [padding_value] * (max_label_length - len(ctc_label))
    return ctc_label

