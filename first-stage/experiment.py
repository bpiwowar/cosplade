from functools import partial
import logging

from attr import Factory
from datamaestro import prepare_dataset
from datamaestro_text.data.conversation.orconvqa import OrConvQADataset

import xpmir.measures as m
from xpmir.conversation.learning import DatasetConversationEntrySampler
from xpmir.conversation.learning.reformulation import DecontextualizedQueryConverter
from xpmir.conversation.models.cosplade import (
    AsymetricMSEContextualizedRepresentationLoss,
    CoSPLADE,
)
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
from xpmir.papers.helpers.optim import TransformerOptimization
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
class Configuration(NeuralIRExperiment):
    """Experimental configuration"""

    splade_model_id: str = "naver/splade-cocondenser-ensembledistil"
    """Base HF id for SPLADE models"""

    launchers: Launchers = Factory(Launchers)
    """Launchers"""

    retrieval_topK: int = 1000
    """How many documents to retrieve"""

    optimization: TransformerOptimization = Factory(TransformerOptimization)
    """Optimization strategy"""

    max_indexed: int = 0
    """Maximum number of indexed documents (debug)"""

    history_max_len: int = 256
    """Maximum length for each history entry"""

    history_size = 0
    """Maximum number of past queries to take into account"""

    queries_max_len: int = 368
    """Maximum length for queries"""


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
        #     prepare_dataset("irds.trec-cast.v1.2019"), MEASURES
        # ),
        cast_2020=Evaluations(
            prepare_dataset("irds.trec-cast.v1.2020.judged"), MEASURES
        ),
        # cast_2021=Evaluations(  # TODO: needs to use passages for 2021
        #     prepare_dataset("irds.trec-cast.v2.2021"), MEASURES
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

    # --- Learn CoSPLADE (1st stage)

    orConvQA: OrConvQADataset = prepare_dataset(
        "com.github.prdwb.orconvqa.preprocessed"
    )
    sampler = DatasetConversationEntrySampler.C(dataset=orConvQA.train)

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
    cosplade = CoSPLADE.C(
        history_size=cfg.history_size,
        history_encoder=history_encoder,
        queries_encoder=queries_encoder,
    ).tag("model", "cosplade")

    trainer = AlignmentTrainer.C(
        sampler=sampler,
        target_model=splade_gold_encoder,
        batcher=PowerAdaptativeBatcher(),
        losses={
            "mse": MSEAlignmentLoss.C(),
            "amse": AsymetricMSEContextualizedRepresentationLoss.C(),
        },
    )

    learner = Learner.C(
        random=cfg.random,
        model=cosplade,
        max_epochs=cfg.optimization.max_epochs,
        optimizers=cfg.optimization.optimizer,
        trainer=trainer,
        use_fp16=True,
        device=device,
        listeners=[],
    )
    output = learner.submit(launcher=learner_launcher)  # type: LearnerOutput
    helper.tensorboard_service.add(learner, learner.logpath)

    # --- Evaluate CoSPLADE

    tests.evaluate_retriever(
        partial(retriever, "cosplade", cosplade),
        init_tasks=[output.learned_model],
        launcher=splade_retriever_launcher,
    )

    # --- Return results
    return PaperResults(
        models={"cosplade-RR@10": cosplade},
        evaluations=tests,
        tb_logs={"cosplade-RR@10": learner.logpath},
    )
