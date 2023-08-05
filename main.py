from CNN_Classifier import logger
from CNN_Classifier.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline

# Data Ingestion

STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>> STAGE: {STAGE_NAME} Started <<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>> STAGE: {STAGE_NAME} Ended Successfully <<<<")
except Exception as e:
    raise e
