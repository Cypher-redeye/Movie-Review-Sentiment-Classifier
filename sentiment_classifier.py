# ====================================================================
# Movie Review Sentiment Classifier (PyTorch Edition)
# Complete Pipeline: Data → Preprocessing → Model → Evaluation
# ====================================================================
#
# Binary sentiment classification (positive/negative) for movie reviews
# using NLTK preprocessing and a Bidirectional LSTM (PyTorch).
#
# Dataset: Stanford IMDB 50K Movie Reviews (auto-downloaded on first run)
# Architecture: Embedding → SpatialDropout → BiLSTM → Dense → Sigmoid
#
# ⚠️  TRAINING TIME WARNING:
#     - GPU (CUDA): ~5-10 min for 15 epochs
#     - CPU only:   ~30-60+ min for 15 epochs (50K reviews, BiLSTM)
#     Set FAST_MODE = True below if you need quick results (reduces data
#     and epochs for a ~5 min CPU run, at the cost of some accuracy).
# ====================================================================

# ====================================================================
# CONFIGURATION — Edit these before running
# ====================================================================

FAST_MODE = False
# True  → Uses 20% of data, 5 epochs, batch_size=128 (~5 min on CPU)
# False → Full dataset, 5 epochs, batch_size=64  (best accuracy)

# ====================================================================
# SECTION 1: IMPORTS & SETUP
# ====================================================================

import os
import re
import sys
import json
import pickle
import tarfile
import urllib.request
import warnings
import string
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')           # Use interactive backend for plots
import matplotlib.pyplot as plt
import seaborn as sns

# --- NLTK Setup ---
import nltk
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet',
                 'omw-1.4', 'averaged_perceptron_tagger']:
    nltk.download(resource, quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- PyTorch ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)

# --- Reproducibility ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Device selection (GPU if available) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU detected — training will run on CPU.")
    if not FAST_MODE:
        print("   Consider setting FAST_MODE = True for a quicker run (~5 min).")

# --- Matplotlib style ---
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
warnings.filterwarnings('ignore')

print(f"\n{'='*60}")
print(f" Sentiment Classifier Pipeline (PyTorch)")
print(f" Mode: {'⚡ FAST (reduced data & epochs)' if FAST_MODE else '🔬 FULL (50K reviews, 15 epochs)'}")
print(f" Device: {device}")
print(f"{'='*60}\n")


# ====================================================================
# SECTION 2: DATA LOADING
# ====================================================================
# Downloads the Stanford IMDB 50K Movie Reviews dataset on first run.
# The raw reviews contain HTML tags (<br />), punctuation, mixed case —
# perfect for demonstrating the full preprocessing pipeline.
# ====================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
IMDB_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'


def download_imdb_dataset(data_dir=DATA_DIR):
    """Download and extract the Stanford IMDB dataset if not already cached."""
    imdb_dir = os.path.join(data_dir, 'aclImdb')
    tar_path = os.path.join(data_dir, 'aclImdb_v1.tar.gz')

    if os.path.isdir(imdb_dir):
        print("📂 IMDB dataset already downloaded — using cached copy.")
    else:
        os.makedirs(data_dir, exist_ok=True)
        print("⬇️  Downloading IMDB dataset (~80 MB)... ", end='', flush=True)
        urllib.request.urlretrieve(IMDB_URL, tar_path)
        print("done.")
        print("📦 Extracting... ", end='', flush=True)
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        os.remove(tar_path)
        print("done.")

    # Parse raw review files into a DataFrame
    reviews, sentiments = [], []
    for split in ['train', 'test']:
        for label in ['pos', 'neg']:
            folder = os.path.join(imdb_dir, split, label)
            for fname in sorted(os.listdir(folder)):
                if fname.endswith('.txt'):
                    with open(os.path.join(folder, fname),
                              'r', encoding='utf-8') as f:
                        reviews.append(f.read())
                        sentiments.append(1 if label == 'pos' else 0)

    df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


print("— Section 2: Loading Data —")
df = download_imdb_dataset()

# In FAST_MODE, use only 20% of the data for speed
if FAST_MODE:
    df = df.sample(frac=0.2, random_state=SEED).reset_index(drop=True)
    print(f"⚡ FAST_MODE: Using {len(df)} reviews (20% subset)")

print(f"   Total reviews : {len(df)}")
print(f"   Positive      : {(df['sentiment'] == 1).sum()}")
print(f"   Negative      : {(df['sentiment'] == 0).sum()}")
print(f"   Sample review : {df['review'].iloc[0][:120]}...\n")


# ====================================================================
# SECTION 3: PREPROCESSING (NLTK)
# ====================================================================
# Pipeline: lowercase → strip HTML/URLs → remove punctuation/digits →
#           tokenize → remove stopwords (KEEP negation words) → lemmatize
# ====================================================================

print("— Section 3: Preprocessing —")

# Build stopword set, but KEEP negation words (they flip sentiment)
NEGATION_WORDS = {'not', 'no', 'nor', 'never', "n't", 'neither',
                  'nobody', 'nothing', 'nowhere', 'hardly', 'scarcely',
                  'barely', "don't", "doesn't", "didn't", "won't",
                  "wouldn't", "couldn't", "shouldn't", "isn't", "aren't",
                  "wasn't", "weren't", "haven't", "hasn't", "hadn't",
                  "can't", "cannot", "mustn't", "needn't"}

stop_words = set(stopwords.words('english'))
# Remove negations from the stopword list so they are preserved
stop_words -= NEGATION_WORDS

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Full NLTK preprocessing pipeline for a single review.

    Steps:
        1. Lowercase
        2. Remove HTML tags (e.g., <br />, <p>)
        3. Remove URLs (http/https/www links)
        4. Remove punctuation and digits
        5. Tokenize with nltk.word_tokenize
        6. Remove stopwords (but keep negation words)
        7. Lemmatize each token (WordNetLemmatizer)
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 4. Remove punctuation and digits (keep apostrophes)
    text = re.sub(r'[^a-z\s\']', ' ', text)

    # 5. Tokenize
    tokens = text.split()
    tokens = [t.strip("'") for t in tokens if t.strip("'")]

    # 6. Remove stopwords (negation words are kept)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # 7. Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


# --- Show before/after on 3 sample reviews ---
print("\n   ┌─────────────────────────────────────────────────────────┐")
print("   │        PREPROCESSING: Before / After Examples          │")
print("   └─────────────────────────────────────────────────────────┘\n")

sample_indices = [0, 1, 2]
for i, idx in enumerate(sample_indices, 1):
    raw = df['review'].iloc[idx]
    cleaned = preprocess_text(raw)
    label = "Positive ✅" if df['sentiment'].iloc[idx] == 1 else "Negative ❌"
    print(f"   ── Example {i} ({label}) ──")
    print(f"   BEFORE: {raw[:200]}...")
    print(f"   AFTER : {cleaned[:200]}...")
    print()

# Apply preprocessing to all reviews
print("   Preprocessing all reviews... ", end='', flush=True)
df['cleaned'] = df['review'].apply(preprocess_text)
print("done.\n")


# ====================================================================
# SECTION 3.5: TRAIN / TEST SPLIT
# ====================================================================
# 80% train, 20% test, stratified to maintain class balance.
# ====================================================================

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'].values,
    df['sentiment'].values,
    test_size=0.2,
    random_state=SEED,
    stratify=df['sentiment'].values
)

print(f"   Train size: {len(X_train)}  |  Test size: {len(X_test)}")
print(f"   Train pos/neg: {sum(y_train)}/{len(y_train)-sum(y_train)}")
print(f"   Test  pos/neg: {sum(y_test)}/{len(y_test)-sum(y_test)}\n")


# ====================================================================
# SECTION 4: TOKENIZATION & SEQUENCING
# ====================================================================
# - Custom vocabulary built from TRAINING data only (no data leakage)
# - num_words=10000 (top 10K most frequent words)
# - OOV token for unseen words at inference time
# - Pad/truncate to maxlen=200 (justified by histogram below)
# ====================================================================

print("— Section 4: Tokenization & Sequencing —")

VOCAB_SIZE  = 10_000
MAX_LEN     = 200
PAD_TOKEN   = '<PAD>'
OOV_TOKEN   = '<OOV>'


class TextTokenizer:
    """
    Custom tokenizer that mimics Keras Tokenizer behavior.
    Builds a vocabulary from training data with a fixed vocab size,
    converts texts to integer sequences, and handles OOV words.
    """

    def __init__(self, num_words=10_000, oov_token='<OOV>'):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}           # word → index
        self.index_word = {}           # index → word
        self.word_counts = {}          # word → frequency

    def fit_on_texts(self, texts):
        """Build vocabulary from a list of text strings."""
        # Count word frequencies
        for text in texts:
            for word in text.split():
                self.word_counts[word] = self.word_counts.get(word, 0) + 1

        # Sort by frequency (descending) and take top num_words - 2
        # (reserving index 0 for PAD and 1 for OOV)
        sorted_words = sorted(self.word_counts.items(),
                              key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:self.num_words - 2]

        # Build word ↔ index mappings
        self.word_index[PAD_TOKEN] = 0
        self.word_index[self.oov_token] = 1
        for idx, (word, _) in enumerate(top_words, start=2):
            self.word_index[word] = idx

        self.index_word = {v: k for k, v in self.word_index.items()}
        print(f"   Vocabulary size: {len(self.word_index)} "
              f"(capped at {self.num_words})")

    def texts_to_sequences(self, texts):
        """Convert list of text strings to list of integer sequences."""
        oov_idx = self.word_index[self.oov_token]
        sequences = []
        for text in texts:
            seq = [self.word_index.get(word, oov_idx)
                   for word in text.split()]
            sequences.append(seq)
        return sequences


def pad_sequences_custom(sequences, maxlen, padding='post',
                         truncating='post', pad_value=0):
    """
    Pad/truncate sequences to a fixed length (mirrors Keras pad_sequences).

    Args:
        sequences: List of integer lists
        maxlen: Target sequence length
        padding: 'pre' or 'post' — where to add padding zeros
        truncating: 'pre' or 'post' — which end to truncate
        pad_value: Value to use for padding (0 = PAD token)

    Returns:
        numpy array of shape (len(sequences), maxlen)
    """
    result = np.full((len(sequences), maxlen), pad_value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        # Truncate if needed
        if len(seq) > maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:
                seq = seq[-maxlen:]
        # Pad
        if padding == 'post':
            result[i, :len(seq)] = seq
        else:
            result[i, -len(seq):] = seq
    return result


# Fit tokenizer on training data only (prevents data leakage)
tokenizer = TextTokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(X_train)

# Convert texts → integer sequences
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences  = tokenizer.texts_to_sequences(X_test)

# --- Sequence-length histogram to justify maxlen ---
train_lengths = [len(seq) for seq in train_sequences]
print(f"   Sequence length stats (training data):")
print(f"     Mean   : {np.mean(train_lengths):.0f}")
print(f"     Median : {np.median(train_lengths):.0f}")
print(f"     95th %%: {np.percentile(train_lengths, 95):.0f}")
print(f"     Max    : {max(train_lengths)}")
coverage = (np.array(train_lengths) <= MAX_LEN).mean() * 100
print(f"   → maxlen={MAX_LEN} covers ~{coverage:.1f}% "
      f"of reviews without truncation.\n")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(train_lengths, bins=80, color='#4C72B0', edgecolor='white', alpha=0.85)
ax.axvline(x=MAX_LEN, color='#C44E52', linestyle='--', linewidth=2,
           label=f'maxlen = {MAX_LEN}')
ax.set_xlabel('Sequence Length (tokens)', fontsize=12)
ax.set_ylabel('Number of Reviews', fontsize=12)
ax.set_title('Distribution of Review Lengths (Training Set)', fontsize=14,
             fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('sequence_length_histogram.png', dpi=150)
# plt.show()

# Pad sequences
X_train_pad = pad_sequences_custom(train_sequences, maxlen=MAX_LEN,
                                   padding='post', truncating='post')
X_test_pad  = pad_sequences_custom(test_sequences, maxlen=MAX_LEN,
                                   padding='post', truncating='post')

print(f"   Padded shapes → Train: {X_train_pad.shape}, Test: {X_test_pad.shape}\n")


# ====================================================================
# SECTION 4.5: PyTorch DATASET & DATALOADER
# ====================================================================

class ReviewDataset(Dataset):
    """PyTorch Dataset for padded review sequences + labels."""

    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# Split training data into train + validation (90/10)
val_size = int(0.1 * len(X_train_pad))
train_size = len(X_train_pad) - val_size

# Shuffle indices for the split
indices = np.random.permutation(len(X_train_pad))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = ReviewDataset(X_train_pad[train_indices],
                              y_train[train_indices])
val_dataset   = ReviewDataset(X_train_pad[val_indices],
                              y_train[val_indices])
test_dataset  = ReviewDataset(X_test_pad, y_test)

BATCH_SIZE = 128 if FAST_MODE else 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=False)

print(f"   DataLoaders ready — batch_size={BATCH_SIZE}")
print(f"   Train batches: {len(train_loader)}, "
      f"Val batches: {len(val_loader)}, "
      f"Test batches: {len(test_loader)}\n")


# ====================================================================
# SECTION 5: MODEL ARCHITECTURE
# ====================================================================
# Embedding(10000, 128) → SpatialDropout(0.2)
# → Bidirectional LSTM(64) → Dense(64, relu) + Dropout(0.3)
# → Dense(1, sigmoid)
# ====================================================================

print("— Section 5: Model Architecture —\n")

EMBEDDING_DIM = 128


class SentimentBiLSTM(nn.Module):
    """
    Bidirectional LSTM for binary sentiment classification.

    Architecture (mirrors the Keras spec):
        Embedding(10000, 128)
        → SpatialDropout1D(0.2)   [drops entire embedding channels]
        → Bidirectional LSTM(64)  [128 output units: 64 forward + 64 backward]
        → Dense(64, ReLU) + Dropout(0.3)
        → Dense(1, Sigmoid)
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim=64,
                 dropout=0.2, recurrent_dropout=0.2):
        super(SentimentBiLSTM, self).__init__()

        # Embedding layer: learns dense vectors for each word
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0          # index 0 = PAD token
        )

        # Spatial dropout: drops entire embedding channels
        # (more effective for NLP than standard dropout on individual neurons)
        self.spatial_dropout = nn.Dropout2d(p=dropout)

        # Bidirectional LSTM: reads sequence forwards AND backwards
        # Output dim = hidden_dim * 2 (64 forward + 64 backward = 128)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=0               # single-layer, so no inter-layer dropout
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 64)   # BiLSTM outputs 128 → 64
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 1)                 # Binary output

    def forward(self, x):
        # x shape: (batch_size, seq_len)

        # Embedding: (batch, seq_len) → (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # Spatial Dropout: reshape for Dropout2d (expects 4D input)
        # (batch, seq_len, embed) → (batch, embed, seq_len, 1) → dropout
        embedded = embedded.unsqueeze(3)
        embedded = embedded.permute(0, 2, 1, 3)
        embedded = self.spatial_dropout(embedded)
        embedded = embedded.permute(0, 2, 1, 3).squeeze(3)
        # Back to (batch, seq_len, embedding_dim)

        # BiLSTM: (batch, seq_len, embed) → (batch, seq_len, hidden*2)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use the last hidden state from both directions
        # hidden shape: (2, batch, hidden_dim) → concat → (batch, hidden*2)
        hidden_fwd = hidden[0]     # Forward LSTM final hidden
        hidden_bwd = hidden[1]     # Backward LSTM final hidden
        combined = torch.cat((hidden_fwd, hidden_bwd), dim=1)

        # Dense layers
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)

        return out.squeeze(1)


# Instantiate model and move to device
model = SentimentBiLSTM(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=64,
    dropout=0.2,
    recurrent_dropout=0.2
).to(device)

# --- Print model summary ---
print("   Model Architecture:")
print("   " + "─" * 55)
total_params = 0
for name, param in model.named_parameters():
    num_params = param.numel()
    total_params += num_params
    print(f"   {name:40s}  {str(list(param.shape)):20s}  "
          f"{num_params:>10,}")
print("   " + "─" * 55)
print(f"   {'Total parameters':40s}  {'':20s}  {total_params:>10,}")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   {'Trainable parameters':40s}  {'':20s}  {trainable:>10,}")
print()

# Loss function and optimizer
criterion = nn.BCELoss()                    # Binary cross-entropy
optimizer = optim.Adam(model.parameters())  # Adam optimizer


# ====================================================================
# SECTION 6: TRAINING
# ====================================================================
# - EarlyStopping: stops if val_loss doesn't improve for 3 epochs,
#   restores the best weights automatically
# - Saves the best model checkpoint to disk
# - validation_split=0.1: uses 10% of training data for validation
#
# ⚠️  TRAINING TIME:
#   GPU: ~5-10 min  |  CPU: ~30-60 min  (full mode)
#   Set FAST_MODE = True at the top of this script for ~5 min on CPU
# ====================================================================

print("— Section 6: Training —")
print(f"   {'⚡ FAST_MODE' if FAST_MODE else '🔬 FULL MODE'}: "
      f"{'5 epochs, batch=128' if FAST_MODE else '15 epochs, batch=64'}\n")

EPOCHS   = 5 if FAST_MODE else 5
PATIENCE = 3

# Training history for plotting later
history = {
    'loss': [], 'accuracy': [],
    'val_loss': [], 'val_accuracy': []
}

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None
MODEL_PATH = 'best_model.pt'

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # ---- Training phase ----
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_seqs, batch_labels in train_loader:
        batch_seqs = batch_seqs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_seqs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_seqs.size(0)
        preds = (outputs >= 0.5).float()
        train_correct += (preds == batch_labels).sum().item()
        train_total += batch_seqs.size(0)

    avg_train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    # ---- Validation phase ----
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_seqs, batch_labels in val_loader:
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_seqs)
            loss = criterion(outputs, batch_labels)

            val_loss += loss.item() * batch_seqs.size(0)
            preds = (outputs >= 0.5).float()
            val_correct += (preds == batch_labels).sum().item()
            val_total += batch_seqs.size(0)

    avg_val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    # Record history
    history['loss'].append(avg_train_loss)
    history['accuracy'].append(train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_accuracy'].append(val_acc)

    epoch_time = time.time() - epoch_start

    print(f"   Epoch {epoch:2d}/{EPOCHS}  │  "
          f"loss: {avg_train_loss:.4f}  acc: {train_acc:.4f}  │  "
          f"val_loss: {avg_val_loss:.4f}  val_acc: {val_acc:.4f}  │  "
          f"{epoch_time:.1f}s", end='')

    # ---- Early stopping + model checkpoint ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        # Save checkpoint to disk
        torch.save(best_model_state, MODEL_PATH)
        print("  ← saved best", end='')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n\n   ⏹  Early stopping triggered (no improvement "
                  f"for {PATIENCE} epochs).")
            break
    print()

# Restore best weights
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"   ✅ Restored best model weights (val_loss={best_val_loss:.4f})")

total_time = time.time() - start_time
print(f"   ⏱  Total training time: {total_time/60:.1f} minutes\n")


# ====================================================================
# SECTION 7: EVALUATION
# ====================================================================
# - Test accuracy, precision, recall, F1-score
# - Confusion matrix (seaborn heatmap)
# - Training vs validation accuracy/loss curves
# - ROC curve + AUC
# ====================================================================

print("— Section 7: Evaluation —\n")

# --- 7a. Predictions on test set ---
model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for batch_seqs, batch_labels in test_loader:
        batch_seqs = batch_seqs.to(device)
        outputs = model(batch_seqs)
        all_probs.extend(outputs.cpu().numpy())
        all_labels.extend(batch_labels.numpy())

y_pred_prob = np.array(all_probs)
y_test_arr  = np.array(all_labels)
y_pred      = (y_pred_prob >= 0.5).astype(int)

# --- 7b. Classification metrics ---
test_acc  = accuracy_score(y_test_arr, y_pred)
test_prec = precision_score(y_test_arr, y_pred)
test_rec  = recall_score(y_test_arr, y_pred)
test_f1   = f1_score(y_test_arr, y_pred)

print(f"   Test Accuracy  : {test_acc:.4f}")
print(f"   Precision      : {test_prec:.4f}")
print(f"   Recall         : {test_rec:.4f}")
print(f"   F1-Score       : {test_f1:.4f}")
print(f"\n   Classification Report:\n")
print(classification_report(y_test_arr, y_pred,
                            target_names=['Negative', 'Positive']))

# --- 7c. Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test_arr, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            annot_kws={'size': 16}, linewidths=0.5, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
# plt.show()

# --- 7d. Training vs Validation Accuracy & Loss Curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, len(history['accuracy']) + 1)

# Accuracy curve
axes[0].plot(epochs_range, history['accuracy'], label='Train Accuracy',
             linewidth=2, marker='o', markersize=5)
axes[0].plot(epochs_range, history['val_accuracy'], label='Val Accuracy',
             linewidth=2, marker='s', markersize=5)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Loss curve
axes[1].plot(epochs_range, history['loss'], label='Train Loss',
             linewidth=2, marker='o', markersize=5)
axes[1].plot(epochs_range, history['val_loss'], label='Val Loss',
             linewidth=2, marker='s', markersize=5)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
# plt.show()

# --- 7e. ROC Curve + AUC ---
fpr, tpr, thresholds = roc_curve(y_test_arr, y_pred_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#4C72B0', linewidth=2.5,
        label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
        label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.15, color='#4C72B0')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
# plt.show()

print(f"   ROC AUC: {roc_auc:.4f}\n")


# ====================================================================
# SECTION 8: INFERENCE
# ====================================================================
# predict_sentiment(text) takes raw text, runs it through the same
# preprocessing → tokenizer → padding pipeline, and returns the
# predicted label with a confidence score.
# ====================================================================

print("— Section 8: Inference —\n")


def predict_sentiment(text, model=model, tokenizer=tokenizer,
                      max_len=MAX_LEN, device=device):
    """
    Predict sentiment for a raw text review.

    Args:
        text (str): Raw review text (can contain HTML, punctuation, etc.)
        model: Trained PyTorch model
        tokenizer: Fitted TextTokenizer
        max_len: Sequence padding length
        device: torch device (cpu/cuda)

    Returns:
        dict with 'label' ("Positive"/"Negative") and 'confidence' (0-1)
    """
    # 1. Run through the same NLTK preprocessing pipeline
    cleaned = preprocess_text(text)

    # 2. Tokenize with the fitted tokenizer
    sequence = tokenizer.texts_to_sequences([cleaned])

    # 3. Pad to the same length used during training
    padded = pad_sequences_custom(sequence, maxlen=max_len,
                                  padding='post', truncating='post')

    # 4. Convert to PyTorch tensor
    tensor = torch.LongTensor(padded).to(device)

    # 5. Predict
    model.eval()
    with torch.no_grad():
        prob = model(tensor).item()

    label = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else 1 - prob

    return {'label': label, 'confidence': float(confidence)}


# --- Test on 5 custom example sentences (edit these!) ---
test_reviews = [
    "This movie was absolutely fantastic! The acting was superb and the "
    "plot kept me on the edge of my seat the entire time.",

    "Terrible waste of time. The script was awful, the acting was wooden, "
    "and I couldn't wait for it to end.",

    "It was okay, nothing special. Some parts were interesting but overall "
    "it felt like a mediocre effort.",

    "I have never seen such a brilliant piece of cinema. The director's "
    "vision was executed flawlessly. A true masterpiece!",

    "Do NOT watch this movie. The plot makes no sense, the characters are "
    "one-dimensional, and the ending is the worst I've ever seen."
]

print("   ┌─────────────────────────────────────────────────────────┐")
print("   │            INFERENCE: Custom Review Predictions         │")
print("   └─────────────────────────────────────────────────────────┘\n")

for i, review in enumerate(test_reviews, 1):
    result = predict_sentiment(review)
    emoji = "✅" if result['label'] == "Positive" else "❌"
    bar_len = int(result['confidence'] * 20)
    bar = '█' * bar_len + '░' * (20 - bar_len)
    print(f"   Review {i}: \"{review[:70]}...\"")
    print(f"   → {emoji} {result['label']}  |  Confidence: "
          f"{result['confidence']:.1%}  [{bar}]")
    print()


# ====================================================================
# SECTION 9: SAVE MODEL & TOKENIZER
# ====================================================================
# Save trained model (.pt) and tokenizer (.pkl) so the pipeline can
# be reloaded without retraining.
#
# To reload later:
#   model = SentimentBiLSTM(10000, 128)
#   model.load_state_dict(torch.load('best_model.pt'))
#   with open('tokenizer.pkl', 'rb') as f:
#       tokenizer = pickle.load(f)
# ====================================================================

print("— Section 9: Saving Artifacts —\n")

# Save the tokenizer (pickle)
TOKENIZER_PATH = 'tokenizer.pkl'
with open(TOKENIZER_PATH, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"   💾 Tokenizer saved → {TOKENIZER_PATH}")

# The best model was already saved during training,
# but let's confirm it's there
if os.path.exists(MODEL_PATH):
    print(f"   💾 Model saved    → {MODEL_PATH}")
else:
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"   💾 Model saved    → {MODEL_PATH}")

print(f"""
{'='*60}
 ✅  Pipeline Complete!
{'='*60}

 Saved files:
   • best_model.pt   — Trained BiLSTM model (PyTorch)
   • tokenizer.pkl   — Fitted tokenizer (vocab + mappings)

 Generated plots:
   • sequence_length_histogram.png
   • confusion_matrix.png
   • training_curves.png
   • roc_curve.png

 To reload and predict without retraining:
   ┌──────────────────────────────────────────────────────────┐
   │  import torch, pickle                                    │
   │                                                          │
   │  model = SentimentBiLSTM(10000, 128)                     │
   │  model.load_state_dict(torch.load('best_model.pt'))      │
   │  model.eval()                                            │
   │                                                          │
   │  with open('tokenizer.pkl', 'rb') as f:                  │
   │      tokenizer = pickle.load(f)                          │
   │                                                          │
   │  result = predict_sentiment("Great movie!", model,       │
   │                              tokenizer)                  │
   └──────────────────────────────────────────────────────────┘
""")
