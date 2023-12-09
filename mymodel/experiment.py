import logging

import xpmir.evaluation
from xpmir.papers.results import PaperResults
from configuration import MyModel
from xpmir.experiments.ir import ir_experiment, ExperimentHelper

logging.basicConfig(level=logging.INFO)


@ir_experiment()
def run(helper: ExperimentHelper, cfg: MyModel) -> PaperResults:
    """My model"""

    # A model that can be loaded, i.e. what is
    # returned by a learner or one of its validation listener.
    my_model = ...

    # The submitted learner
    learner = ...

    # The results of evaluations
    tests: xpmir.evaluation.EvaluationsCollection = ...

    return PaperResults(
        models={"MyModel-RR@10": my_model},
        evaluations=tests,
        tb_logs={"MyModel-RR@10": learner.logpath},
    )
