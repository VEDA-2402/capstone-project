"""
Main Pipeline Execution
main.py
"""
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.utils.common import logger, print_section

from pathlib import Path

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)


def run_data_ingestion():
    """Execute data ingestion pipeline"""
    try:
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        
        data_ingestion = DataIngestion(config=data_ingestion_config)
        ingested_data_path = data_ingestion.initiate_data_ingestion()
        
        return ingested_data_path
    
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise e


def run_data_preprocessing():
    """Execute data preprocessing pipeline"""
    try:
        config_manager = ConfigurationManager()
        preprocessing_config = config_manager.get_data_preprocessing_config()
        
        data_preprocessing = DataPreprocessing(config=preprocessing_config)
        preprocessed_data_path = data_preprocessing.initiate_data_preprocessing()
        
        return preprocessed_data_path
    
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise e


def run_model_training():
    """Execute model training pipeline"""
    # TODO: Implement after model training component is ready
    logger.info("Model training will be implemented next...")
    pass


def run_model_evaluation():
    """Execute model evaluation pipeline"""
    # TODO: Implement after model evaluation component is ready
    logger.info("Model evaluation will be implemented next...")
    pass


def main():
    """Main pipeline execution"""
    try:
        print_section("SENTIMENT ANALYSIS PIPELINE - STARTING")
        
        # Stage 1: Data Ingestion
        logger.info("\n>>> STAGE 1: DATA INGESTION <<<")
        ingested_data_path = run_data_ingestion()
        logger.info(f"✓ Stage 1 completed: {ingested_data_path}")
        
        # Stage 2: Data Preprocessing
        logger.info("\n>>> STAGE 2: DATA PREPROCESSING <<<")
        preprocessed_data_path = run_data_preprocessing()
        logger.info(f"✓ Stage 2 completed: {preprocessed_data_path}")
        
        # Stage 3: Model Training
        logger.info("\n>>> STAGE 3: MODEL TRAINING <<<")
        # run_model_training()
        logger.info("✓ Stage 3 - To be implemented")
        
        # Stage 4: Model Evaluation
        logger.info("\n>>> STAGE 4: MODEL EVALUATION <<<")
        # run_model_evaluation()
        logger.info("✓ Stage 4 - To be implemented")
        
        print_section("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("All stages executed successfully!")
        logger.info(f"\nOutput files:")
        logger.info(f"  - Ingested data: {ingested_data_path}")
        logger.info(f"  - Preprocessed data: {preprocessed_data_path}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()