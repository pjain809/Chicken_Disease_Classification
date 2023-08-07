from CNN_Classifier import logger
from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.prepare_callbacks import PrepareCallbacks
from CNN_Classifier.components.training import Training

STAGE_NAME = "Model Training"


class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callbacks_config()
        prepare_callbacks = PrepareCallbacks(prepare_callbacks_config)
        callbacks_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(callback_list=callbacks_list)


if __name__ == "__main__":
    try:
        logger.info(f">>>> STAGE: {STAGE_NAME} Started <<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>> STAGE: {STAGE_NAME} Ended Successfully <<<<\nx==================================x")
    except Exception as e:
        logger.exception(e)
        raise e
