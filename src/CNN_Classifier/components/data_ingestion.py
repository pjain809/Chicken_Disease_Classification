import os
import zipfile
from pathlib import Path
import urllib.request as request
from CNN_Classifier import logger
from CNN_Classifier.utils.common import get_size
from CNN_Classifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Downloads the Chicken-Disease Data from URL
        """
        if not(os.path.exists(self.config.local_data_file)):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded with following headers:\n{headers}")
        else:
            logger.info(f"File: {self.config.local_data_file} already exists "
                        f"(Size: {get_size(Path(self.config.local_data_file))})")

    def extract_zip_file(self):
        """
        Extracts the ZIP File at the Unzip Path
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(Path(unzip_path), exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
