"""
Configuration file for the sentiment analysis pipeline
src/config/configuration.py
"""
import os
from dataclasses import dataclass
from pathlib import Path


class DataIngestionConfig:
    """Configuration for data ingestion"""
    def __init__(self):
        self.raw_data_path = Path("data/raw/all-data.csv")
        self.ingested_data_path = Path("data/processed/ingested_data.csv")
        self.column_names = ['Sentiment', 'Text']
        self.encoding = "latin-1"
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.ingested_data_path), exist_ok=True)


class DataPreprocessingConfig:
    """Configuration for data preprocessing"""
    def __init__(self):
        self.ingested_data_path = Path("data/processed/ingested_data.csv")
        self.preprocessed_data_path = Path("data/processed/preprocessed_data.csv")
        
        # Artifact paths
        self.artifacts_dir = Path("data/artifacts")
        self.vocab_path = Path("data/artifacts/vocab.pkl")
        self.label_encoder_path = Path("data/artifacts/label_encoder.pkl")
        self.max_len_path = Path("data/artifacts/max_len.pkl")
        
        # Preprocessing parameters
        self.min_word_frequency = 2
        self.max_len_percentile = 95
        
        # Create directories if they don't exist
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.preprocessed_data_path), exist_ok=True)


class ModelTrainingConfig:
    """Configuration for model training"""
    def __init__(self):
        self.preprocessed_data_path = Path("data/processed/preprocessed_data.csv")
        self.model_path = Path("data/artifacts/model.pkl")
        self.train_test_split_ratio = 0.2
        self.random_state = 42
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)


class ModelEvaluationConfig:
    """Configuration for model evaluation"""
    def __init__(self):
        self.model_path = Path("data/artifacts/model.pkl")
        self.evaluation_results_path = Path("data/artifacts/evaluation_results.pkl")
        
        os.makedirs(os.path.dirname(self.evaluation_results_path), exist_ok=True)


class ConfigurationManager:
    """Centralized configuration manager"""
    
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()
        self.model_training_config = ModelTrainingConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
    
    def get_data_ingestion_config(self):
        return self.data_ingestion_config
    
    def get_data_preprocessing_config(self):
        return self.data_preprocessing_config
    
    def get_model_training_config(self):
        return self.model_training_config
    
    def get_model_evaluation_config(self):
        return self.model_evaluation_config