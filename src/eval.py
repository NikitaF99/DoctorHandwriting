
import torch
from utility import *
from model import CRNN
import rapidfuzz.distance.Levenshtein as Levenshtein
from torch.utils.data import  DataLoader
from config import *
from main import create_dataset
import pandas as pd
# from data_utility import preprocess_data, create_dataset, create_dataloaders
from preprocess import preprocess_data
from dataset import  create_dataset, create_dataloaders
import matplotlib.pyplot as plt
import json

def load_data_params():
    df_train = pd.read_csv('../dataset/Training/training_labels.csv')
    df_test = pd.read_csv('../dataset/Testing/testing_labels.csv')
    df_val = pd.read_csv('../dataset/Validation/validation_labels.csv')

    
    df_train, df_test, df_val = preprocess_data(df_train, df_test, df_val)

    all_labels = pd.concat([df_train['processed_label'], df_test['processed_label'], df_val['processed_label']])
    all_chars = sorted(list(set(''.join(all_labels.astype(str).tolist()))))

    char_to_int = {char: i for i, char in enumerate(all_chars)}
    int_to_char = {i: char for i, char in enumerate(all_chars)}
    max_label_length = max(all_labels.astype(str).apply(len))
    num_classes = len(char_to_int) + 1
 
    lexicon = set()
    lexicon.update(df_train['processed_label'])
    lexicon.update(df_test['processed_label'])
    lexicon.update(df_val['processed_label'])
    lexicon = sorted(lexicon)

    data = {"INT_TO_CHAR": int_to_char,
            "NUM_OF_CLASSES": num_classes,
            "LEXICON": list(lexicon),
            "CHAR_TO_INT": char_to_int,
            "MAX_LABEL_LENGTH": max_label_length}
    
    # with open("parameters.json", "w") as f:
    #     json.dump(data, f, indent=4)

   
    return char_to_int, int_to_char, max_label_length, num_classes, df_test, lexicon

def evaluation(test_dataloader, num_classes, device, int_to_char, char_to_int, lexicon=None):
    model = CRNN(num_classes=num_classes).to(device)   # initialize same architecture as before
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    test_decoded_predictions = []
    test_true_labels = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass to get the model outputs (log probabilities)
            outputs = model(images) 

            decoded_predictions = ctc_decode(outputs, int_to_char, lexicon=lexicon) # Pass the lexicon
            test_decoded_predictions.extend(decoded_predictions)

            # Convert true labels back to strings for comparison
            true_labels_list = []
            for label_tensor in labels:
                true_label_chars = [int_to_char.get(l.item(), '') for l in label_tensor if l.item() != len(char_to_int)]
                true_labels_list.append(''.join(true_label_chars))
            test_true_labels.extend(true_labels_list)

 
    print("\nExample Predictions vs. True Labels:")
    for i in range(min(10, len(test_decoded_predictions))):
        print(f"True: {test_true_labels[i]}, Predicted: {test_decoded_predictions[i]}")

    # CER
    cer = character_error_rate(test_true_labels, test_decoded_predictions)
    print(f"\nCharacter Error Rate (CER) on Test Set: {cer:.4f}")

    
    # WER

    wer = word_error_rate(test_true_labels, test_decoded_predictions)
    print(f"\nWord Error Rate (WER) on Test Set: {wer:.4f}")

    metrics = ['WER', 'CER']
    values = [wer, cer]

    visualize(metrics, values)

def visualize(metrics, values):

    plt.figure(figsize=(6, 4))
    plt.bar(metrics, values, color=['blue', 'red'])
    plt.ylabel('Error Rate')
    plt.title('Word Error Rate (WER) and Character Error Rate (CER)')
    plt.ylim(0, 1) # Error rates are  between 0 and 1
    plt.show()

def main():
    char_to_int, int_to_char, max_label_length, num_classes, df_test, lexicon = load_data_params()
    dataset = create_dataset(df_test, char_to_int=char_to_int, max_label_length=max_label_length)
    dataloader =  DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(lexicon)
    evaluation(dataloader, num_classes, 'cpu', int_to_char, char_to_int, lexicon)


if __name__ == "__main__":
    main()

