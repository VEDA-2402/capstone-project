"""
Common utility functions for the sentiment analysis pipeline
src/utils/common.py
"""

import logging
import os
import pickle
from pathlib import Path

# ==================== LOGGER SETUP ====================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    handlers=[
        logging.FileHandler("logs/running_logs.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("sentiment_analysis")


# ==================== UTILITY FUNCTIONS ====================

def print_section(title):
    """
    Print a formatted section header
    
    Args:
        title: Section title to display
    """
    separator = "=" * 70
    logger.info(f"\n{separator}")
    logger.info(f"{title.center(70)}")
    logger.info(f"{separator}\n")


def file_exists(file_path):
    """
    Check if a file exists
    
    Args:
        file_path: Path to the file (str or Path object)
    
    Returns:
        bool: True if file exists, False otherwise
    """
    return Path(file_path).exists()


def save_pickle(obj, file_path):
    """
    Save object as pickle file
    
    Args:
        obj: Object to save
        file_path: Path where to save the pickle file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save object
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        
        logger.info(f"✓ Saved pickle file: {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving pickle file {file_path}: {str(e)}")
        raise e


def load_pickle(file_path):
    """
    Load object from pickle file
    
    Args:
        file_path: Path to the pickle file
    
    Returns:
        Loaded object
    """
    try:
        if not file_exists(file_path):
            raise FileNotFoundError(f"Pickle file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        
        logger.info(f"✓ Loaded pickle file: {file_path}")
        return obj
        
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {str(e)}")
        raise e

