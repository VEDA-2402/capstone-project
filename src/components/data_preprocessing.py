"""
Data Preprocessing Component
src/components/data_preprocessing.py
"""
import os
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from src.config.configuration import DataPreprocessingConfig
from src.utils.common import logger, print_section, file_exists, save_pickle


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        """
        Initialize Data Preprocessing component
        
        Args:
            config: DataPreprocessingConfig object containing paths and settings
        """
        self.config = config
        self.vocab = None
        self.label_encoder = None
        self.max_len = None
        logger.info("DataPreprocessing component initialized")
    
    def load_ingested_data(self):
        """
        Load ingested data from previous step
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            print_section("LOADING INGESTED DATA")
            
            if not file_exists(self.config.ingested_data_path):
                raise FileNotFoundError(f"Ingested data not found at: {self.config.ingested_data_path}")
            
            df = pd.read_csv(self.config.ingested_data_path)
            logger.info(f"Loaded data shape: {df.shape}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading ingested data: {str(e)}")
            raise e
    
    def clean_data(self, df):
        """
        Clean the dataset by removing null values
        
        Args:
            df: pandas DataFrame
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        try:
            print_section("DATA CLEANING")
            
            initial_shape = df.shape
            df = df.dropna()
            final_shape = df.shape
            
            logger.info(f"Initial shape: {initial_shape}")
            logger.info(f"After dropping nulls: {final_shape}")
            logger.info(f"Rows removed: {initial_shape[0] - final_shape[0]}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error during cleaning: {str(e)}")
            raise e
    
    @staticmethod
    def preprocess_text(text):
        """
        Clean and preprocess individual text
        
        Args:
            text: Input text string
            
        Returns:
            str: Cleaned text
        """
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def apply_text_preprocessing(self, df):
        """
        Apply text preprocessing to entire dataset
        
        Args:
            df: pandas DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with cleaned text
        """
        try:
            print_section("TEXT PREPROCESSING")
            
            # Apply preprocessing
            df['Clean_Text'] = df['Text'].apply(self.preprocess_text)
            
            # Remove empty texts
            initial_len = len(df)
            df = df[df['Clean_Text'].str.len() > 0]
            final_len = len(df)
            
            logger.info(f"After preprocessing: {df.shape}")
            logger.info(f"Empty texts removed: {initial_len - final_len}")
            
            # Show example
            logger.info(f"\nExample - Original:")
            logger.info(f"{df['Text'].iloc[0]}")
            logger.info(f"\nExample - Cleaned:")
            logger.info(f"{df['Clean_Text'].iloc[0]}")
            
            # Calculate word statistics
            df['word_count'] = df['Clean_Text'].str.split().str.len()
            logger.info(f"\nWord count statistics:")
            logger.info(f"\n{df['word_count'].describe()}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error during text preprocessing: {str(e)}")
            raise e
    
    def encode_labels(self, df):
        """
        Encode sentiment labels to numeric values
        
        Args:
            df: pandas DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with encoded labels
        """
        try:
            print_section("LABEL ENCODING")
            
            self.label_encoder = LabelEncoder()
            df['label'] = self.label_encoder.fit_transform(df['Sentiment'])
            
            logger.info("\nLabel mapping:")
            for idx, sentiment in enumerate(self.label_encoder.classes_):
                logger.info(f"  {sentiment}: {idx}")
            
            logger.info(f"\nLabel distribution:")
            logger.info(f"\n{df['label'].value_counts().sort_index()}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error during label encoding: {str(e)}")
            raise e
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from texts with minimum frequency threshold
        
        Args:
            texts: List or array of text strings
            
        Returns:
            tuple: (vocab dict, word_counts Counter)
        """
        try:
            print_section("VOCABULARY BUILDING")
            
            word_counts = Counter()
            
            for text in texts:
                word_counts.update(text.split())
            
            # Special tokens
            vocab = {'<PAD>': 0, '<UNK>': 1}
            idx = 2
            
            # Add words that meet minimum frequency
            for word, count in word_counts.most_common():
                if count >= self.config.min_word_frequency:
                    vocab[word] = idx
                    idx += 1
            
            logger.info(f"Vocabulary size: {len(vocab)}")
            logger.info(f"Total unique words: {len(word_counts)}")
            logger.info(f"Minimum word frequency threshold: {self.config.min_word_frequency}")
            
            logger.info("\nTop 15 most common words:")
            for word, count in word_counts.most_common(15):
                logger.info(f"  {word}: {count}")
            
            self.vocab = vocab
            return vocab, word_counts
        
        except Exception as e:
            logger.error(f"Error building vocabulary: {str(e)}")
            raise e
    
    def determine_max_length(self, df):
        """
        Determine maximum sequence length based on percentile
        
        Args:
            df: pandas DataFrame with word_count column
            
        Returns:
            int: Maximum sequence length
        """
        try:
            print_section("SEQUENCE LENGTH ANALYSIS")
            
            percentiles = [50, 75, 90, 95, 99]
            logger.info("Word count percentiles:")
            for p in percentiles:
                val = np.percentile(df['word_count'], p)
                logger.info(f"  {p}th percentile: {val:.0f} words")
            
            # Set MAX_LEN to specified percentile
            self.max_len = int(np.percentile(df['word_count'], self.config.max_len_percentile))
            logger.info(f"\nSet MAX_LEN to {self.config.max_len_percentile}th percentile: {self.max_len}")
            
            return self.max_len
        
        except Exception as e:
            logger.error(f"Error determining max length: {str(e)}")
            raise e
    
    def text_to_sequence(self, text):
        """
        Convert text to sequence of indices with padding/truncation
        
        Args:
            text: Input text string
            
        Returns:
            list: Sequence of token indices
        """
        words = text.split()
        sequence = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # Pad or truncate
        if len(sequence) < self.max_len:
            sequence += [self.vocab['<PAD>']] * (self.max_len - len(sequence))
        else:
            sequence = sequence[:self.max_len]
        
        return sequence
    
    def convert_to_sequences(self, df):
        """
        Convert all texts to sequences
        
        Args:
            df: pandas DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with sequences
        """
        try:
            print_section("TEXT TO SEQUENCE CONVERSION")
            
            df['sequence'] = df['Clean_Text'].apply(self.text_to_sequence)
            
            logger.info(f"Sequence length: {self.max_len}")
            logger.info(f"\nExample sequence (first 15 tokens):")
            logger.info(f"{df['sequence'].iloc[0][:15]}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error converting to sequences: {str(e)}")
            raise e
    
    def save_artifacts(self):
        """
        Save all preprocessing artifacts (vocab, encoder, max_len)
        """
        try:
            print_section("SAVING PREPROCESSING ARTIFACTS")
            
            # Save vocabulary
            save_pickle(self.vocab, self.config.vocab_path)
            
            # Save label encoder
            save_pickle(self.label_encoder, self.config.label_encoder_path)
            
            # Save max_len
            save_pickle(self.max_len, self.config.max_len_path)
            
            logger.info("✓ All artifacts saved successfully")
        
        except Exception as e:
            logger.error(f"Error saving artifacts: {str(e)}")
            raise e
    
    def save_preprocessed_data(self, df):
        """
        Save preprocessed data
        
        Args:
            df: pandas DataFrame
        """
        try:
            print_section("SAVING PREPROCESSED DATA")
            
            # Select relevant columns
            df_to_save = df[['Sentiment', 'Clean_Text', 'label', 'sequence']]
            
            # Save to CSV
            df_to_save.to_csv(self.config.preprocessed_data_path, index=False)
            logger.info(f"✓ Preprocessed data saved to: {self.config.preprocessed_data_path}")
            logger.info(f"✓ Total records: {len(df_to_save)}")
        
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {str(e)}")
            raise e
    
    def print_summary(self, df):
        """
        Print preprocessing summary
        
        Args:
            df: pandas DataFrame
        """
        print_section("PREPROCESSING COMPLETE - SUMMARY")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Vocabulary size: {len(self.vocab)}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        logger.info(f"Classes: {list(self.label_encoder.classes_)}")
        logger.info(f"Sequence length (MAX_LEN): {self.max_len}")
        logger.info(f"\nLabel distribution:")
        logger.info(f"\n{df['label'].value_counts().sort_index()}")
        logger.info("\n✓ Data is ready for train-test split and model training!")
    
    def initiate_data_preprocessing(self):
        """
        Main method to execute the complete preprocessing pipeline
        
        Returns:
            str: Path to preprocessed data file
        """
        try:
            print_section("STARTING DATA PREPROCESSING PIPELINE")
            
            # Step 1: Load ingested data
            df = self.load_ingested_data()
            
            # Step 2: Clean data
            df = self.clean_data(df)
            
            # Step 3: Preprocess text
            df = self.apply_text_preprocessing(df)
            
            # Step 4: Encode labels
            df = self.encode_labels(df)
            
            # Step 5: Build vocabulary
            self.build_vocabulary(df['Clean_Text'].values)
            
            # Step 6: Determine max sequence length
            self.determine_max_length(df)
            
            # Step 7: Convert texts to sequences
            df = self.convert_to_sequences(df)
            
            # Step 8: Save artifacts
            self.save_artifacts()
            
            # Step 9: Save preprocessed data
            self.save_preprocessed_data(df)
            
            # Step 10: Print summary
            self.print_summary(df)
            
            print_section("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
            logger.info(f"Output file: {self.config.preprocessed_data_path}")
            
            return str(self.config.preprocessed_data_path)
        
        except Exception as e:
            logger.error(f"Data preprocessing pipeline failed: {str(e)}")
            raise e


if __name__ == "__main__":
    # Test the component independently
    from src.config.configuration import ConfigurationManager
    
    config_manager = ConfigurationManager()
    preprocessing_config = config_manager.get_data_preprocessing_config()
    
    data_preprocessing = DataPreprocessing(config=preprocessing_config)
    preprocessed_data_path = data_preprocessing.initiate_data_preprocessing()
    
    print(f"\n✓ Preprocessed data available at: {preprocessed_data_path}")