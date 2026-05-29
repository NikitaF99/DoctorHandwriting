import os
import pandas as pd
# Enable CPU fallback for MPS kernels that are not yet implemented (e.g. CTC loss)
# Must be set before importing torch to take effect.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import cv2
from utility import get_device
from config import *
from dataset import PrescriptionDataset
# from data_utility import preprocess_data, create_dataset, create_dataloaders
from preprocess import preprocess_data
from dataset import  create_dataset, create_dataloaders
from model import CRNN, EarlyStopping
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from utility import *
import json

def main():
    # Device
    device = get_device()
    print("Using device:", device)
    df_train = pd.read_csv('../dataset/Training/training_labels.csv')
    df_test = pd.read_csv('../dataset/Testing/testing_labels.csv')
    df_val = pd.read_csv('../dataset/Validation/validation_labels.csv')

    
    df_train, df_test, df_val = preprocess_data(df_train, df_test, df_val)

    all_labels = pd.concat([df_train['processed_label'], df_test['processed_label'], df_val['processed_label']])
    all_chars = sorted(list(set(''.join(all_labels.astype(str).tolist()))))
    char_to_int = {char: i for i, char in enumerate(all_chars)}
    int_to_char = {i: char for i, char in enumerate(all_chars)}
    
    INT_TO_CHAR = int_to_char
    max_label_length = max(all_labels.astype(str).apply(len))

    train_dataset = create_dataset(df_train, char_to_int=char_to_int, max_label_length=max_label_length)
    test_dataset = create_dataset(df_test, char_to_int=char_to_int, max_label_length=max_label_length)
    val_dataset = create_dataset(df_val, char_to_int=char_to_int, max_label_length=max_label_length)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Training DataLoader created.")
    print("Validation DataLoader created.")
    print("Testing DataLoader created.")


    # Instantiate the CRNN model with the corrected architecture
    num_classes = len(char_to_int) + 1
    model = CRNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CTCLoss(blank=len(char_to_int)) # Use the size of the character set as the blank index
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    config = {
    "num_classes": num_classes,
    "max_label_length": int(max_label_length),
    "char_to_int": char_to_int,
    "int_to_char": int_to_char
    }

    with open("crnn_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    # Implement the training loop
    num_epochs = EPOCHS 
    train_losses = []
    val_losses = []
    cer_scores = []
    wer_scores = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                # Get the actual output width from the CNN for the current batch size
                temp = model.cnn(images)
                _, _, _, output_width = temp.size()
                input_lengths = torch.full((images.size(0),), output_width, dtype=torch.long).to(device)

            target_lengths = torch.tensor([len([l for l in label if l != len(char_to_int)]) for label in labels], dtype=torch.long).to(device)


            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            flat_labels = torch.cat([label[:target_lengths[i]] for i, label in enumerate(labels)])
            loss = criterion(outputs, flat_labels, input_lengths, target_lengths)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item() # Accumulate loss

        epoch_loss = running_loss / len(train_dataloader) # Calculate average loss per batch
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Validation step
        model.eval()
        val_true_labels = []
        val_predicted_labels = []
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                # Get input lengths
                temp = model.cnn(images)
                _, _, _, output_width = temp.size()
                input_lengths = torch.full((images.size(0),), output_width, dtype=torch.long).to(device)

                # Get target lengths
                target_lengths = torch.tensor([len([l for l in label if l != len(char_to_int)]) for label in labels], dtype=torch.long).to(device)

                outputs = model(images)

                flat_labels = torch.cat([label[:target_lengths[i]] for i, label in enumerate(labels)])

                loss = criterion(outputs, flat_labels, input_lengths, target_lengths)

                val_running_loss += loss.item() # Accumulate loss

                log_probs = F.log_softmax(outputs, dim=2) 

                decoded_texts = ctc_decode(log_probs.cpu(), int_to_char, lexicon=None, beam_width=5)

        # Convert ground truth back to strings
                for label in labels:
                    label = label[label != len(char_to_int)]  # remove padding
                    text = ''.join([int_to_char[int(i)] for i in label.cpu().numpy()])
                    val_true_labels.append(text)

                val_predicted_labels.extend(decoded_texts)

        val_epoch_loss = val_running_loss / len(val_dataloader) # Calculate average loss per batch
        train_losses.append(epoch_loss)
        val_losses.append(val_epoch_loss)
        val_cer = character_error_rate(val_true_labels, val_predicted_labels)
        val_wer = word_error_rate(val_true_labels, val_predicted_labels)
        cer_scores.append(val_cer)
        wer_scores.append(val_wer)

     

        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}")

        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    data = {
            "TRAIN_LOSS": train_losses,
            "VAL_LOSS": val_losses,
            "CER": cer_scores,
            "WER": wer_scores,
    }
    
    with open("vals.json", "w") as f:
        json.dump(data, f, indent=4)

    display_graph(train_losses, val_losses, cer_scores, wer_scores)

    print("Training finished.")
    early_stopping.load_best_model(model)
    
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved")


def display_graph(train_losses, val_losses, cer_scores, wer_scores):
    plt.figure(figsize=(12, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('CTC Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    # CER/WER
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cer_scores)+1), cer_scores, label='CER', marker='x')
    plt.plot(range(1, len(wer_scores)+1), wer_scores, label='WER', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.title('CER and WER over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()



if __name__ == "__main__":
    main()