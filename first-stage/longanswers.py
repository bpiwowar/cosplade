from typing import Iterator
from experimaestro import Constant, Param
from datamaestro_text.data.conversation import ConversationDataset
from datamaestro.record import Record
from xpmir.learning import Random

from datamaestro_text.data.conversation.base import (
    AnswerEntry,
    ConversationTree,
    EntryType,
    RetrievedEntry,
    SingleConversationTree,
)


class LongAnswersAdapter(ConversationDataset):
    source: Param[ConversationDataset]
    random: Param[Random]
    id: Constant[str] = ""

    def __iter__(self) -> Iterator[ConversationTree]:
        random = self.random.state

        for entry in self.source:
            history = [entry.history[0]]

            for h_entry in entry.history[1:]:
                if h_entry[EntryType] == EntryType.SYSTEM_ANSWER:
                    if (e_answer := h_entry.get(AnswerEntry)) and (
                        retrieved := h_entry.get(RetrievedEntry)
                    ):
                        answer = e_answer.answer
                        text = e_answer.answer
                        for doc in retrieved.documents:
                            if answer in doc:
                                sentences = doc.split(".")
                                if len(sentences) > 3:
                                    index = random.randint(0, len(sentences) - 2)
                                    text = ".".join(sentences[index : index + 3]) + "."
                                else:
                                    text = doc
                                break

                        h_entry = Record(
                            AnswerEntry(text),
                            EntryType.SYSTEM_ANSWER,
                        )
                history.append(h_entry)

            yield SingleConversationTree(entry.id, history)
