# === Import Required Libraries ===
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import pickle

# === 1. Load Dataset ===
df = pd.read_csv('all-data.csv', names=['Sentiment', 'Text'], encoding='latin-1')

print("="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nSentiment distribution:")
print(df['Sentiment'].value_counts())

# === 2. Data Cleaning ===
print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

# Drop missing values
df = df.dropna()
print(f"After dropping nulls: {df.shape}")

# === 3. Text Preprocessing ===
print("\n" + "="*60)
print("TEXT PREPROCESSING")
print("="*60)

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply preprocessing
df['Clean_Text'] = df['Text'].apply(preprocess_text)

# Remove empty texts
df = df[df['Clean_Text'].str.len() > 0]

print(f"After preprocessing: {df.shape}")
print(f"\nExample - Original:")
print(df['Text'].iloc[0])
print(f"\nExample - Cleaned:")
print(df['Clean_Text'].iloc[0])

# === 4. Text Statistics ===
df['word_count'] = df['Clean_Text'].str.split().str.len()

print(f"\nWord count statistics:")
print(df['word_count'].describe())

# === 5. Label Encoding ===
print("\n" + "="*60)
print("LABEL ENCODING")
print("="*60)

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Sentiment'])

print("\nLabel mapping:")
for idx, sentiment in enumerate(label_encoder.classes_):
    print(f"  {sentiment}: {idx}")

# === 6. Build Vocabulary ===
print("\n" + "="*60)
print("VOCABULARY BUILDING")
print("="*60)

def build_vocabulary(texts, min_freq=2):
    """Build vocabulary from texts with minimum frequency threshold"""
    word_counts = Counter()
    
    for text in texts:
        word_counts.update(text.split())
    
    # Special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    
    # Add words that meet minimum frequency
    for word, count in word_counts.most_common():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab, word_counts

vocab, word_counts = build_vocabulary(df['Clean_Text'].values, min_freq=2)

print(f"Vocabulary size: {len(vocab)}")
print(f"Total unique words: {len(word_counts)}")

print("\nTop 15 most common words:")
for word, count in word_counts.most_common(15):
    print(f"  {word}: {count}")

# === 7. Determine Maximum Sequence Length ===
print("\n" + "="*60)
print("SEQUENCE LENGTH ANALYSIS")
print("="*60)

# Calculate percentiles
percentiles = [50, 75, 90, 95, 99]
print("Word count percentiles:")
for p in percentiles:
    val = np.percentile(df['word_count'], p)
    print(f"  {p}th percentile: {val:.0f} words")

# Set MAX_LEN to 95th percentile
MAX_LEN = int(np.percentile(df['word_count'], 95))
print(f"\nRecommended MAX_LEN: {MAX_LEN}")

# === 8. Text to Sequence Conversion ===
print("\n" + "="*60)
print("TEXT TO SEQUENCE CONVERSION")
print("="*60)

def text_to_sequence(text, vocab, max_len):
    """Convert text to sequence of indices with padding/truncation"""
    words = text.split()
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words]
    
    # Pad or truncate
    if len(sequence) < max_len:
        sequence += [vocab['<PAD>']] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    
    return sequence

# Apply conversion
df['sequence'] = df['Clean_Text'].apply(lambda x: text_to_sequence(x, vocab, MAX_LEN))

print(f"Sequence length: {MAX_LEN}")
print(f"\nExample sequence (first 15 tokens):")
print(df['sequence'].iloc[0][:15])

# === 9. Save Processed Data ===
print("\n" + "="*60)
print("SAVING PREPROCESSED DATA")
print("="*60)

# Save vocabulary
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save MAX_LEN
with open('max_len.pkl', 'wb') as f:
    pickle.dump(MAX_LEN, f)

# Save processed dataframe
df[['Sentiment', 'Clean_Text', 'label', 'sequence']].to_csv('preprocessed_data.csv', index=False)

print("✓ Saved vocab.pkl")
print("✓ Saved label_encoder.pkl")
print("✓ Saved max_len.pkl")
print("✓ Saved preprocessed_data.csv")

# === 10. Final Summary ===
print("\n" + "="*60)
print("PREPROCESSING COMPLETE - SUMMARY")
print("="*60)
print(f"Total samples: {len(df)}")
print(f"Vocabulary size: {len(vocab)}")
print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"Classes: {list(label_encoder.classes_)}")
print(f"Sequence length (MAX_LEN): {MAX_LEN}")
print(f"\nLabel distribution:")
print(df['label'].value_counts().sort_index())
print("\n✓ Data is ready for train-test split and model training!")