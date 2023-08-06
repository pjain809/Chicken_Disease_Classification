from CNN_Classifier import logger
from CNN_Classifier.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from CNN_Classifier.pipeline.stage02_prepare_base_model import PrepareBaseModelTrainingPipeline
from CNN_Classifier.pipeline.stage03_prepare_callbacks import PrepareCallbacksTrainingPipeline


# Data Ingestion
STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>> STAGE: {STAGE_NAME} Started <<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>> STAGE: {STAGE_NAME} Ended Successfully <<<<\nx==================================x")
except Exception as e:
    logger.exception(e)
    raise e


# Prepare Base Model
STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f">>>> STAGE: {STAGE_NAME} Started <<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>> STAGE: {STAGE_NAME} Ended Successfully <<<<\nx==================================x")
except Exception as e:
    logger.exception(e)
    raise e


# Prepare Callbacks
STAGE_NAME = "Prepare Callbacks"

try:
    logger.info(f">>>> STAGE: {STAGE_NAME} Started <<<<")
    obj = PrepareCallbacksTrainingPipeline()
    obj.main()
    logger.info(f">>>> STAGE: {STAGE_NAME} Ended Successfully <<<<\nx==================================x")
except Exception as e:
    logger.exception(e)
    raise e
