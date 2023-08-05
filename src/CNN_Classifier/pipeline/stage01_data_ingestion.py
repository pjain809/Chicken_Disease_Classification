from CNN_Classifier import logger
from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info(f">>>> STAGE: {STAGE_NAME} Started <<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>> STAGE: {STAGE_NAME} Ended Successfully <<<<")
    except Exception as e:
        raise e
