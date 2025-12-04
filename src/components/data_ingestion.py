"""
Data Ingestion Component
src/components/data_ingestion.py
"""
import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config.configuration import DataIngestionConfig
from src.utils.common import logger, print_section, file_exists


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize Data Ingestion component
        
        Args:
            config: DataIngestionConfig object containing paths and settings
        """
        self.config = config
        logger.info("DataIngestion component initialized")
    
    def load_data(self):
        """
        Load raw data from CSV file
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            print_section("DATA INGESTION - LOADING RAW DATA")
            
            # Check if raw data exists
            if not file_exists(self.config.raw_data_path):
                raise FileNotFoundError(f"Raw data not found at: {self.config.raw_data_path}")
            
            # Load data
            df = pd.read_csv(
                self.config.raw_data_path,
                names=self.config.column_names,
                encoding=self.config.encoding
            )
            
            logger.info(f"Dataset loaded successfully")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise e
    
    def validate_data(self, df):
        """
        Perform basic validation checks on the data
        
        Args:
            df: pandas DataFrame
            
        Returns:
            bool: True if validation passes
        """
        try:
            print_section("DATA VALIDATION")
            
            # Check if dataframe is empty
            if df.empty:
                logger.error("Dataset is empty!")
                return False
            
            # Check for required columns
            required_columns = self.config.column_names
            missing_columns = set(required_columns) - set(df.columns)
            
            if missing_columns:
                logger.error(f"Missing columns: {missing_columns}")
                return False
            
            # Check data types
            logger.info(f"\nData types:")
            logger.info(f"\n{df.dtypes}")
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            logger.info(f"\nMissing values:")
            logger.info(f"\n{missing_counts}")
            
            # Display sentiment distribution
            logger.info(f"\nSentiment distribution:")
            logger.info(f"\n{df['Sentiment'].value_counts()}")
            
            # Display sample data
            logger.info(f"\nFirst 3 rows:")
            logger.info(f"\n{df.head(3)}")
            
            logger.info("✓ Data validation passed")
            return True
        
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return False
    
    def save_ingested_data(self, df):
        """
        Save ingested data to processed directory
        
        Args:
            df: pandas DataFrame to save
        """
        try:
            print_section("SAVING INGESTED DATA")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.ingested_data_path), exist_ok=True)
            
            # Save data
            df.to_csv(self.config.ingested_data_path, index=False)
            logger.info(f"✓ Data saved to: {self.config.ingested_data_path}")
            logger.info(f"✓ Total records saved: {len(df)}")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise e
    
    def initiate_data_ingestion(self):
        """
        Main method to execute the complete data ingestion pipeline
        
        Returns:
            str: Path to ingested data file
        """
        try:
            print_section("STARTING DATA INGESTION PIPELINE")
            
            # Step 1: Load data
            df = self.load_data()
            
            # Step 2: Validate data
            if not self.validate_data(df):
                raise ValueError("Data validation failed")
            
            # Step 3: Save ingested data
            self.save_ingested_data(df)
            
            print_section("DATA INGESTION COMPLETED SUCCESSFULLY")
            logger.info(f"Output file: {self.config.ingested_data_path}")
            
            return str(self.config.ingested_data_path)
        
        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {str(e)}")
            raise e


if __name__ == "__main__":
    # Test the component independently
    from src.config.configuration import ConfigurationManager
    
    config_manager = ConfigurationManager()
    data_ingestion_config = config_manager.get_data_ingestion_config()
    
    data_ingestion = DataIngestion(config=data_ingestion_config)
    ingested_data_path = data_ingestion.initiate_data_ingestion()
    
    print(f"\n✓ Ingested data available at: {ingested_data_path}")
