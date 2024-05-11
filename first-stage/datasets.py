from functools import lru_cache
from datamaestro_text.data.ir import Adhoc, Topics, DocumentStore, TopicRecord
from experimaestro import Param
from typing import Iterator, Type
from datamaestro_text.data.ir import TextItem
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
