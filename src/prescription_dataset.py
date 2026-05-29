import torch
from torch.utils.data import Dataset
import pandas as pd


class PrescriptionDataset(Dataset):
    def __init__(self, df, target_height=32, max_width=256, char_to_int=None, max_label_length=None):
        self.df = df
        self.target_height = target_height
        self.max_width = max_width
        self.char_to_int = char_to_int
        self.max_label_length = max_label_length


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Assuming 'preprocessed_image_crnn' contains the preprocessed tensor and 'ctc_label' contains the CTC label list
        image_tensor = self.df.iloc[idx]['preprocessed_image_crnn']
        ctc_label = self.df.iloc[idx]['ctc_label']

        # Convert label list to tensor
        ctc_label_tensor = torch.tensor(ctc_label, dtype=torch.long)

        return image_tensor, ctc_label_tensor

