from typing import Iterator
from experimaestro import Constant, Param
from datamaestro_text.data.conversation import ConversationDataset
from datamaestro.record import Record
import numpy as np
from xpmir.learning import Random

from datamaestro_text.data.conversation.base import (
    AnswerEntry,
    ConversationTree,
    EntryType,
    RetrievedEntry,
    SingleConversationTree,
)


def lcs(S, T):
    m = len(S)
    n = len(T)
    counter = [[0] * (n + 1) for x in range(m + 1)]
    longest = 0
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i + 1][j + 1] = c
                if c > longest:
                    longest = c
    return longest


def overlap(short_answer, passage):
    return lcs(short_answer, passage) > 0.8 * len(short_answer)


def trim_answer(
    nlp, random: np.random.RandomState, short_answer, passage, trimmed_len=4
):
    sentences = [s.text for s in nlp(passage).sents]

    for i in range(len(sentences)):
        if short_answer in sentences[i] or (
            len(short_answer) > 30 and overlap(short_answer, sentences[i])
        ):
            sentences = sentences[max(0, i - 3) : min(len(sentences), i + 4)]
            if len(sentences) <= trimmed_len:
                start = 0
            else:
                start = random.choice(len(sentences) - trimmed_len + 1)
            sentences = sentences[
                max(0, start) : min(len(sentences), start + trimmed_len)
            ]
            return " ".join(sentences)
    return None


class LongAnswersAdapter(ConversationDataset):
    source: Param[ConversationDataset]
    random: Param[Random]
    id: Constant[str] = ""

    version: Constant[int] = 2
    """Chagelog:

    - v2: use original segmenting/search code
    """

    def __iter__(self) -> Iterator[ConversationTree]:
        random = self.random.state
        import spacy

        nlp = spacy.load("en_core_web_sm")

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
                            if text := trim_answer(nlp, random, answer, doc):
                                break

                        h_entry = Record(
                            AnswerEntry(text),
                            EntryType.SYSTEM_ANSWER,
                        )
                history.append(h_entry)

            yield SingleConversationTree(entry.id, history)
