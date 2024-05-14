from functools import lru_cache
from datamaestro_text.data.ir import Adhoc, Topics, DocumentStore, TopicRecord
from datamaestro.record import Record
from experimaestro import Param
from typing import Dict, Iterator, List, Type
from xpmir.rankers import Retriever, ScoredDocument
from datamaestro_text.data.ir import TextItem, IDItem
from datamaestro_text.data.conversation.base import (
    ConversationHistoryItem,
    AnswerDocumentID,
    AnswerEntry,
)


class HistoryAnswerHydrator(Topics):
    """Transforms document IDs into their text in conversation history"""

    topics: Param[Topics]
    store: Param[DocumentStore]

    @staticmethod
    def wrap(source: Adhoc):
        return Adhoc(
            id="",
            topics=HistoryAnswerHydrator(
                id="", topics=source.topics, store=source.documents
            ),
            assessments=source.assessments,
            documents=source.documents,
        )

    @property
    def topic_recordtype(self) -> Type[TopicRecord]:
        """The class for topics"""
        return self.topics.topic_recordtype

    @lru_cache(maxsize=2048)
    def get_document(self, doc_id):
        """Use a cache since history can contain many times the same document"""
        return self.store.document_ext(doc_id)[TextItem].text

    def iter(self) -> Iterator[TopicRecord]:
        for record in self.topics:
            history = []
            for h in record[ConversationHistoryItem].history:
                if r_id := h.get(AnswerDocumentID):
                    history.append(
                        h.update(AnswerEntry(self.get_document(r_id.document_id)))
                    )
                else:
                    history.append(h)

            yield record.update(ConversationHistoryItem(history))


class MaxPassageRetriever(Retriever):
    retriever: Param[Retriever]

    topK: Param[int]

    def initialize(self):
        self.retriever.initialize()
        super().initialize()

    def retrieve_all(
        self, queries: Dict[str, Record]
    ) -> Dict[str, List[ScoredDocument]]:
        return {
            key: self.process(value)
            for key, value in super().retrieve_all(queries).items()
        }

    def retrieve(self, record: TopicRecord) -> List[ScoredDocument]:
        return self.process(super().retrieve(record))

    def process(self, scored_documents: List[ScoredDocument]):
        retrieved = set()
        aggregated: List[ScoredDocument] = []

        # Just in case
        scored_documents.sort(reverse=True)

        for sd in scored_documents:
            doc_id, p_id = sd.document[IDItem].id.rsplit("-", 1)
            if doc_id not in retrieved:
                retrieved.add(doc_id)
                aggregated.append(ScoredDocument(Record(IDItem(doc_id)), sd.score))

        return aggregated[: self.topK]
