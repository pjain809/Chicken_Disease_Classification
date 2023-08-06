from CNN_Classifier import logger
from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.prepare_callbacks import PrepareCallbacks

STAGE_NAME = "Prepare Callbacks"


class PrepareCallbacksTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callbacks_config()
        prepare_callbacks = PrepareCallbacks(prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()


if __name__ == "__main__":
    try:
        logger.info(f">>>> STAGE: {STAGE_NAME} Started <<<<")
        obj = PrepareCallbacksTrainingPipeline()
        obj.main()
        logger.info(f">>>> STAGE: {STAGE_NAME} Ended Successfully <<<<\nx==================================x")
    except Exception as e:
        logger.exception(e)
        raise e
