from CNN_Classifier import logger
from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.model_evaluation import Evaluation

STAGE_NAME = "Model Evaluation"


class EvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(evaluation_config)
        evaluation.evaluation()
        evaluation.save_score()


if __name__ == "__main__":
    try:
        logger.info(f">>>> STAGE: {STAGE_NAME} Started <<<<")
        obj = EvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>> STAGE: {STAGE_NAME} Ended Successfully <<<<\nx==================================x")
    except Exception as e:
        logger.exception(e)
        raise e