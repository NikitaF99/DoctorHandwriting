import torch
import random
import numpy as np
import os
import torch
import torch.nn.functional as F
from rapidfuzz import process
import rapidfuzz.distance.Levenshtein as Levenshtein


def get_device():
    """Return the best available device (MPS for Apple Silicon, CUDA for Nvidia, else CPU)."""
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    if torch.backends.mps.is_available():
        print("WARNING: Using MPS with CPU fallback for CTC Loss. Performance may be impacted.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    


def ctc_beam_search_decode(log_probs, int_to_char, beam_width=5, blank_index=None):
    """
    Performs simple beam search decoding for CTC outputs.

    Args:
        log_probs: (time_steps, num_classes) log probabilities tensor for one sample.
        int_to_char: Dictionary mapping integer → character.
        beam_width: Number of beams to keep at each step.
        blank_index: Index representing the CTC blank token.

    Returns:
        A list of (decoded_string, score) tuples, sorted by descending score.
    """
    time_steps, num_classes = log_probs.shape
    blank_index = blank_index if blank_index is not None else num_classes - 1

    # Each beam is a tuple (string, score)
    beams = [("", 0.0)]  # Log probability space

    for t in range(time_steps):
        next_beams = {}
        probs = F.log_softmax(log_probs[t], dim=-1)

        for prefix, score in beams:
            for c in range(num_classes):
                new_score = score + probs[c].item()

                if c == blank_index:
                    # Stay on same prefix (blank means "no new char")
                    next_beams[prefix] = max(next_beams.get(prefix, float("-inf")), new_score)
                else:
                    new_prefix = prefix + int_to_char.get(c, "")
                    next_beams[new_prefix] = max(next_beams.get(new_prefix, float("-inf")), new_score)

        # Keep top N beams only
        beams = sorted(next_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]

    return beams


def ctc_decode(log_probs, int_to_char, lexicon=None, beam_width=5):
    """
    Decodes CRNN output using beam search + lexicon filtering + fuzzy fallback.

    Args:
        log_probs: Tensor (time_steps, batch_size, num_classes)
        int_to_char: Mapping integer → character
        lexicon: List or set of valid words
        beam_width: Number of candidate beams

    Returns:
        List of decoded strings for each item in batch.
    """
    batch_size = log_probs.size(1)
    num_classes = log_probs.size(2)
    blank_index = num_classes - 1

    decoded_labels = []

    for i in range(batch_size):
        beams = ctc_beam_search_decode(log_probs[:, i, :], int_to_char, beam_width, blank_index)
        candidates = [seq for seq, _ in beams]

        # print(f"\nTop-{beam_width} beam candidates for sample {i}: {candidates}")

        # Lexicon filter
        if lexicon:
            lexicon = list(lexicon)
            valid_candidates = [c for c in candidates if c in lexicon]

            if valid_candidates:
                best_candidate = valid_candidates[0]  # Top beam that’s in lexicon
                # print(f"Lexicon match: {best_candidate}")
                decoded_labels.append(best_candidate)
                continue

            # 2️⃣ Fuzzy fallback if no exact lexicon hit
            best_match = process.extractOne(candidates[0], lexicon)
            if best_match:
                # print(f"Fuzzy fallback: {best_match[0]} (score {best_match[1]:.2f})")
                decoded_labels.append(best_match[0])
            else:
                decoded_labels.append(candidates[0])  # No match found, return top beam
        else:
            # No lexicon — just take best beam
            decoded_labels.append(candidates[0])

    return decoded_labels

def character_error_rate(test_true_labels, test_decoded_predictions):
    total_characters = 0
    total_edit_distance = 0

    for true_label, predicted_label in zip(test_true_labels, test_decoded_predictions):
        total_characters += len(true_label)
        total_edit_distance += levenshtein_distance(true_label, predicted_label)


    cer = total_edit_distance / total_characters if total_characters > 0 else 0

    return cer

def word_error_rate(test_true_labels,test_decoded_predictions ):
    # WER
    total_words = 0
    total_word_edit_distance = 0

    for true_label, predicted_label in zip(test_true_labels, test_decoded_predictions):
        true_words = true_label.split(" ")
        predicted_words = predicted_label.split(" ")

        total_words += len(true_words)
        # Using rapidfuzz's distance for word lists
        total_word_edit_distance += Levenshtein.distance(true_words, predicted_words)

    wer = total_word_edit_distance / total_words if total_words > 0 else 0
    return wer


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]