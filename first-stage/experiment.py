from functools import partial
from itertools import product
import logging
from typing import List

from attr import Factory
from datamaestro import prepare_dataset
from datamaestro_text.data.conversation.orconvqa import OrConvQADataset
from experimaestro import tag, tagspath
import xpmir.measures as m
from xpmir.conversation.learning import (
    DatasetConversationEntrySampler,
    DatasetConversationIterator,
)
from xpmir.conversation.learning.reformulation import DecontextualizedQueryConverter
from xpmir.conversation.models.cosplade import (
    AsymetricMSEContextualizedRepresentationLoss,
    CoSPLADE,
)
from xpmir.learning.optim import ParameterOptimizer, RegexParameterFilter, Adam
from xpmir.evaluation import Evaluations, EvaluationsCollection
from xpmir.experiments.helpers import LauncherSpecification, NeuralIRExperiment
from datamaestro_text.data.ir import Adhoc
from xpmir.experiments.ir import IRExperimentHelper, ir_experiment
from xpmir.index.sparse import SparseRetriever, SparseRetrieverIndexBuilder
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.learning.devices import CudaDevice
from xpmir.learning.learner import Learner
from xpmir.learning.trainers.validation import TrainerValidationLoss
from xpmir.letor.trainers.alignment import (
    AlignmentTrainer,
    MSEAlignmentLoss,
    CosineAlignmentLoss,
)
from xpmir.neural.splade import MaxAggregation, SpladeTextEncoderV2
from xpmir.papers import configuration
from xpmir.papers.results import PaperResults
from xpmir.rankers import Documents, Retriever, document_cache
from xpmir.text.huggingface import (
    HFListTokenizer,
    HFStringTokenizer,
    HFTokenizer,
    HFTokenizerAdapter,
)
from xpmir.text.huggingface.base import HFMaskedLanguageModel, HFModelConfigFromId
from xpmir.text.adapters import TopicTextConverter

from longanswers import LongAnswersAdapter
from datasets import HistoryAnswerHydrator, MaxPassageRetriever

logging.basicConfig(level=logging.INFO)


@configuration
class Launchers:
    learner: LauncherSpecification = Factory(
        lambda: LauncherSpecification(requirements="cpu(cores=1) & cuda(mem=8G)")
    )
    splade_indexer: LauncherSpecification = Factory(
        lambda: LauncherSpecification(requirements="cpu(cores=1)  & cuda(mem=8G)")
    )
    splade_retriever: LauncherSpecification = Factory(
        lambda: LauncherSpecification(requirements="cpu(cores=1)  & cuda(mem=8G)")
    )


@configuration()
class Optimization:
    epochs: int = 4
    """Number of epochs"""

    steps_per_epoch: int = 8
    """Steps per epoch"""

    batch_size: int = 8
    """Number of samples per step"""

    queries_lr = 2e-5
    """Learning rate for encoding query history"""

    query_answer_lr = 3e-5
    """Learning rate encoding query/answer"""

    validation_interval: int = 4
    """2 validations"""


@configuration()
class Configuration(NeuralIRExperiment):
    """Experimental configuration"""

    splade_model_id: str = "naver/splade-cocondenser-ensembledistil"
    """Base HF id for SPLADE models"""

    launchers: Launchers = Factory(Launchers)
    """Launchers"""

    retrieval_topK: int = 1000
    """How many documents to retrieve"""

    optimization: Optimization = Factory(Optimization)
    """Optimization strategy"""

    max_indexed: int = 0
    """Maximum number of indexed documents (debug)"""

    history_max_len: int = 256
    """Maximum number of tokens for each query/history entry"""

    queries_max_len: int = 368
    """Maximum number of tokens for queries"""

    history_size: List[int] = [1]
    """Maximum number of past queries to take into account"""


MEASURES = [
    m.R @ 500,
    m.R @ 1000,
    m.AP @ 1000,
    m.AP @ 500,
    m.nDCG @ 3,
    m.nDCG @ 20,
    m.nDCG @ 500,
    m.nDCG @ 1000,
    m.RR @ 10,
]


@ir_experiment()
def run(helper: IRExperimentHelper, cfg: Configuration) -> PaperResults:
    """CoSPLADE training"""

    # --- Get launchers

    device = CudaDevice.C()
    splade_indexer_launcher = cfg.launchers.splade_indexer.launcher
    splade_retriever_launcher = cfg.launchers.splade_retriever.launcher
    learner_launcher = cfg.launchers.learner.launcher

    mp_device = CudaDevice.C(distributed=True)

    # --- SPLADE (from HuggingFace)

    tokenizer = HFTokenizer.C(model_id=cfg.splade_model_id)
    splade_encoder = SpladeTextEncoderV2.C(
        tokenizer=HFStringTokenizer.C(tokenizer=tokenizer),
        encoder=HFMaskedLanguageModel.from_pretrained_id(cfg.splade_model_id),
        aggregation=MaxAggregation.C(),
        maxlen=256,
    )

    # --- Evaluation

    cast_2021 = prepare_dataset("irds.trec-cast.v2.2021")

    tests = EvaluationsCollection(
        # cast_2019=Evaluations(
        #     prepare_dataset("irds.trec-cast.v1.2019.jugded"), MEASURES
        # ),
        cast_2020=Evaluations(
            HistoryAnswerHydrator.wrap(
                prepare_dataset("irds.trec-cast.v1.2020.judged")
            ),
            MEASURES,
        ),
        cast_2021=Evaluations(
            Adhoc(
                id="",
                topics=HistoryAnswerHydrator(
                    id="", topics=cast_2021.topics, store=cast_2021.documents
                ),
                assessments=cast_2021.assessments,
                documents=prepare_dataset("irds.trec-cast.v2.passages.documents"),
            ),
            MEASURES,
        ),
        cast_2022=Evaluations(prepare_dataset("irds.trec-cast.v3.2022"), MEASURES),
    )

    # Caches the Splade index task for a document collection
    @document_cache
    def splade_index(documents: Documents):
        return SparseRetrieverIndexBuilder.C(
            batch_size=512,
            batcher=PowerAdaptativeBatcher(),
            encoder=splade_encoder,
            device=mp_device,
            documents=documents,
            ordered_index=False,
            max_docs=cfg.max_indexed,
        ).submit(launcher=splade_indexer_launcher)

    # --- Evaluate with manually rewritten queries

    splade_gold_encoder = SpladeTextEncoderV2.C(
        tokenizer=HFTokenizerAdapter.C(
            tokenizer=tokenizer, converter=DecontextualizedQueryConverter.C()
        ),
        encoder=HFMaskedLanguageModel.from_pretrained_id(cfg.splade_model_id),
        aggregation=MaxAggregation.C(),
    )
    splade_raw_encoder = SpladeTextEncoderV2.C(
        tokenizer=HFTokenizerAdapter.C(
            tokenizer=tokenizer, converter=TopicTextConverter.C()
        ),
        encoder=HFMaskedLanguageModel.from_pretrained_id(cfg.splade_model_id),
        aggregation=MaxAggregation.C(),
    )

    def retriever(
        name,
        encoder,
        documents: Documents,
    ) -> Retriever.C:
        if documents.id == "irds.trec-cast.v2.passages.documents@irds":
            # Merge results
            base_retriever = SparseRetriever.C(
                index=splade_index()(documents),
                topk=cfg.retrieval_topK * 3,
                batchsize=1,
                encoder=encoder,
                in_memory=False,
                device=device,
            ).tag("model", name)

            return MaxPassageRetriever.C(
                retriever=base_retriever, topK=cfg.retrieval_topK
            )

        return SparseRetriever.C(
            index=splade_index()(documents),
            topk=cfg.retrieval_topK,
            batchsize=1,
            encoder=encoder,
            in_memory=False,
            device=device,
        ).tag("model", name)

    tests.evaluate_retriever(
        partial(retriever, "splade-gold", splade_gold_encoder),
        launcher=splade_retriever_launcher,
    )
    tests.evaluate_retriever(
        partial(retriever, "splade-raw", splade_raw_encoder),
        launcher=splade_retriever_launcher,
    )

    # --- Learn and evaluate CoSPLADE (1st stage)

    models = {}
    tb_logs = {}

    orConvQA: OrConvQADataset = prepare_dataset(
        "com.github.prdwb.orconvqa.preprocessed"
    )

    DATA_TRANSFORMS = {
        "normal": lambda data: data,
        "long": lambda data: LongAnswersAdapter.C(source=data, random=cfg.random),
    }

    LOSSES = {
        "mse+amse": {
            "mse": MSEAlignmentLoss.C(),
            "amse": AsymetricMSEContextualizedRepresentationLoss.C(weight=0.5),
        },
        "cosine": {
            "cosine": CosineAlignmentLoss.C(),
        },
    }

    # Use different learning rates for the query history encoder
    # and the other ones
    optimizers = [
        ParameterOptimizer.C(
            optimizer=Adam.C(lr=cfg.optimization.queries_lr),
            filter=RegexParameterFilter(includes=[r"(^|\.)queries_encoder\."]),
        ),
        ParameterOptimizer.C(
            optimizer=Adam.C(lr=cfg.optimization.query_answer_lr),
        ),
    ]

    def process(cosplade, trainer, validation_data):
        """Learn and evaluate a CoSPLADE model

        :param cosplade: The coSPLADE model
        :param trainer: The trainer
        :param validation_data: The data used for validation
        """
        learner = Learner.C(
            random=cfg.random,
            model=cosplade,
            max_epochs=cfg.optimization.epochs,
            steps_per_epoch=cfg.optimization.steps_per_epoch,
            optimizers=optimizers,
            trainer=trainer,
            use_fp16=True,
            device=device,
            listeners=[
                TrainerValidationLoss.C(
                    id="val",
                    trainer=trainer,
                    data=validation_data,
                    batch_size=256,
                    batcher=PowerAdaptativeBatcher.C(),
                    validation_interval=cfg.optimization.validation_interval,
                )
            ],
        )
        output = learner.submit(launcher=learner_launcher)
        helper.tensorboard_service.add(learner, learner.logpath)

        # --- Evaluate CoSPLADE

        learned_model = output.listeners["val"]

        tests.evaluate_retriever(
            partial(retriever, "cosplade", cosplade),
            init_tasks=[learned_model],
            launcher=splade_retriever_launcher,
        )

        models[tagspath(cosplade)] = learned_model
        tb_logs[tagspath(cosplade)] = learner.logpath

    # --- Test different variants

    history_encoder = SpladeTextEncoderV2.C(
        tokenizer=HFStringTokenizer.C(tokenizer=tokenizer),
        encoder=HFMaskedLanguageModel.C(
            config=HFModelConfigFromId.C(model_id=cfg.splade_model_id)
        ),
        aggregation=MaxAggregation(),
        maxlen=cfg.history_max_len,
    )

    queries_encoder = SpladeTextEncoderV2.C(
        tokenizer=HFListTokenizer.C(tokenizer=tokenizer),
        encoder=HFMaskedLanguageModel.C(
            config=HFModelConfigFromId.C(model_id=cfg.splade_model_id)
        ),
        aggregation=MaxAggregation.C(),
        maxlen=cfg.queries_max_len,
    )

    for (transform_id, data_transform), history_size in product(
        DATA_TRANSFORMS.items(), cfg.history_size
    ):
        cosplade = CoSPLADE.C(
            history_size=tag(history_size),
            history_encoder=history_encoder,
            queries_encoder=queries_encoder,
        ).tag("model", "cosplade")

        # Use train and test for training
        sampler = DatasetConversationEntrySampler.C(
            datasets=[data_transform(orConvQA.train), data_transform(orConvQA.test)]
        ).tag("answers", transform_id)

        # Use validation data to validate
        validation_data = DatasetConversationIterator.C(
            datasets=[data_transform(orConvQA.validation)]
        )

        # Iterate over losses
        for losses_key, losses in LOSSES.items():
            trainer = AlignmentTrainer.C(
                sampler=sampler,
                target_model=splade_gold_encoder,
                batcher=PowerAdaptativeBatcher(),
                losses=losses,
            ).tag("loss", losses_key)

            process(cosplade, trainer, validation_data)

        # Save the models so they can be re-used
        # helper.xp.save(models)

    # --- Return results
    return PaperResults(
        models=models,
        evaluations=tests,
        tb_logs=tb_logs,
    )
