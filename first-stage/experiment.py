from functools import partial
import logging
from typing import List

from attr import Factory
from datamaestro import prepare_dataset
from datamaestro_text.data.conversation.orconvqa import OrConvQADataset
from experimaestro import tag, tagspath
import xpmir.measures as m
from xpmir.conversation.learning import DatasetConversationEntrySampler
from xpmir.conversation.learning.reformulation import DecontextualizedQueryConverter
from xpmir.conversation.models.cosplade import (
    AsymetricMSEContextualizedRepresentationLoss,
    CoSPLADE,
)
from xpmir.learning.optim import ParameterOptimizer, RegexParameterFilter, Adam
from xpmir.evaluation import Evaluations, EvaluationsCollection
from xpmir.experiments.helpers import LauncherSpecification, NeuralIRExperiment
from xpmir.experiments.ir import IRExperimentHelper, ir_experiment
from xpmir.index.sparse import SparseRetriever, SparseRetrieverIndexBuilder
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.learning.devices import CudaDevice
from xpmir.learning.learner import Learner
from xpmir.letor.trainers.alignment import AlignmentTrainer, MSEAlignmentLoss
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

from datasets import HistoryAnswerHydrator

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
    m.R @ 1000,
    m.AP @ 1000,
    m.nDCG @ 3,
    m.nDCG @ 20,
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
        # cast_2021=Evaluations(
        #     HistoryAnswerHydrator.wrap(
        #         prepare_dataset("irds.trec-cast.v2.2021")
        #     ), MEASURES
        # ),
        # cast_2022=Evaluations(
        #     prepare_dataset("irds.trec-cast.v3.2022"), MEASURES
        # ),
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
    sampler = DatasetConversationEntrySampler.C(dataset=orConvQA.train)

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

    def process(cosplade, trainer):
        learner = Learner.C(
            random=cfg.random,
            model=cosplade,
            max_epochs=cfg.optimization.epochs,
            steps_per_epoch=cfg.optimization.steps_per_epoch,
            optimizers=optimizers,
            trainer=trainer,
            use_fp16=True,
            device=device,
            listeners=[],
        )
        output = learner.submit(launcher=learner_launcher)
        helper.tensorboard_service.add(learner, learner.logpath)

        # --- Evaluate CoSPLADE

        tests.evaluate_retriever(
            partial(retriever, "cosplade", cosplade),
            init_tasks=[output.learned_model],
            launcher=splade_retriever_launcher,
        )

        models[tagspath(cosplade)] = cosplade
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

    for history_size in cfg.history_size:
        cosplade = CoSPLADE.C(
            history_size=tag(history_size),
            history_encoder=history_encoder,
            queries_encoder=queries_encoder,
        ).tag("model", "cosplade")

        trainer = AlignmentTrainer.C(
            sampler=sampler,
            target_model=splade_gold_encoder,
            batcher=PowerAdaptativeBatcher(),
            losses={
                "mse": MSEAlignmentLoss.C(),
                "amse": AsymetricMSEContextualizedRepresentationLoss.C(weight=0.5),
            },
        )

        process(cosplade, trainer)

    # --- Return results
    return PaperResults(
        models=models,
        evaluations=tests,
        tb_logs=tb_logs,
    )
