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

    def iter(self) -> Iterator[TopicRecord]:
        for record in self.topics:
            history = []
            for h in record[ConversationHistoryItem].history:
                if r_id := h.get(AnswerDocumentID):
                    document = self.store.document_ext(r_id.document_id)
                    history.append(h.update(AnswerEntry(document[TextItem].text)))
                else:
                    history.append(h)

            yield record.update(ConversationHistoryItem(history))
