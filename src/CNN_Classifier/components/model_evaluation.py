import os

import tensorflow as tf
from pathlib import Path
from CNN_Classifier.utils.common import save_json
from CNN_Classifier.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.model = None
        self.score = None
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(rescale=1./255, validation_split=0.30)
        dataflow_kwargs = dict(target_size=self.config.params_image_size[:-1],
                               batch_size=self.config.params_batch_size,
                               interpolation="bilinear")

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator=valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

    def save_score(self):
        scores = {"Loss": self.score[0], "Accuracy": self.score[1]}
        save_json(Path("artifacts/model_evaluation/Scores.json"), scores)
