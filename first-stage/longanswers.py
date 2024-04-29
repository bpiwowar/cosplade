from datamaestro_text.data.conversation.orconvqa import OrConvQADataset
from datamaestro.record import Record
from experimaestro import Config, Param
from typing import Iterator

from datamaestro_text.data.conversation.base import (
    AnswerEntry,
    ConversationTree,
    EntryType,
    RetrievedEntry,
    SingleConversationTree,
)

import numpy as np

class LongAnswersAdapter(Config):
    source: Param[OrConvQADataset]
    
    def __iter__(self) -> Iterator[ConversationTree]:
        for entry in self.source:
            history = [entry.history[0]]

            for h_entry in entry.history[1:]:
                answer = h_entry[AnswerEntry].answer
                documents = h_entry[RetrievedEntry].documents    
                for doc in documents:
                    if answer in doc:
                        sentences = doc.split(".")
                        index = np.random.randint(0, len(sentences)-2)
                        text = ".".join(sentences[index:index+3]) + "."

                record = Record(
                    AnswerEntry(text),
                    EntryType.SYSTEM_ANSWER,
                )
                history.append(record)
                
            yield SingleConversationTree(entry.id, history)