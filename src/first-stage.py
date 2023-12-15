import logging

from datamaestro import prepare_dataset
from datamaestro_text.data.conversation.canard import CanardDataset
from datamaestro_text.data.conversation.orconvqa import OrConvQADataset
import xpmir.evaluation
from xpmir.papers.results import PaperResults
from xpmir.experiments.ir import ir_experiment, ExperimentHelper
from xpmir.learning.learner import Learner
from configuration import MyModel

logging.basicConfig(level=logging.INFO)


@ir_experiment()
def run(helper: ExperimentHelper, cfg: MyModel) -> PaperResults:
    """My model"""

    # --- Prepare the datasets

    canard: CanardDataset = prepare_dataset("com.github.aagohary.canard")
    orConvQA: OrConvQADataset = prepare_dataset(
        "com.github.prdwb.orconvqa.preprocessed"
    )

    # The submitted learner
    learner = Learner()

    # The results of evaluations
    tests: xpmir.evaluation.EvaluationsCollection = ...

    return PaperResults(
        models={"MyModel-RR@10": my_model},
        evaluations=tests,
        tb_logs={"MyModel-RR@10": learner.logpath},
    )
