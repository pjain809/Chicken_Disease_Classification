from CNN_Classifier import logger
from CNN_Classifier.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from CNN_Classifier.pipeline.stage02_prepare_base_model import PrepareBaseModelTrainingPipeline
from CNN_Classifier.pipeline.stage03_training import TrainingPipeline
from CNN_Classifier.pipeline.stage04_model_evaluation import EvaluationTrainingPipeline


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


# Training
STAGE_NAME = "Model Training"

try:
    logger.info(f">>>> STAGE: {STAGE_NAME} Started <<<<")
    obj = TrainingPipeline()
    obj.main()
    logger.info(f">>>> STAGE: {STAGE_NAME} Ended Successfully <<<<\nx==================================x")
except Exception as e:
    logger.exception(e)
    raise e


# Model Evaluation
STAGE_NAME = "Model Evaluation"

try:
    logger.info(f">>>> STAGE: {STAGE_NAME} Started <<<<")
    obj = EvaluationTrainingPipeline()
    obj.main()
    logger.info(f">>>> STAGE: {STAGE_NAME} Ended Successfully <<<<\nx==================================x")
except Exception as e:
    logger.exception(e)
    raise e