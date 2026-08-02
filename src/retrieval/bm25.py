"""BM25 lexical retrieval."""

from __future__ import annotations
import math
import pickle
import re
from collections import Counter
from config import CHUNKS_PATH, BM25_INDEX_PATH
from src.corpus_io import load_chunks

K1 = 1.5
B = 0.75
# k1 controls how much term freq will affect score
# B controls doc length normalization factor


def text_tokenize(text: str) -> list[str]:
    # this converts text to lowercase so no matching issues
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever():
    def __init__(self, data_chunks):
        self.chunks = data_chunks
        self.list_term_freq = []
        self.doc_freq = Counter()
        self.doc_length = []
        self.avg_doc_length = 0
        self.index_create()

    def index_create(self):
        # this will make inverted index that bm25 will use
        # each chunk is treated as a document for retrieval
        doc_total_length = 0

        for chunk_data in self.chunks:
            tokens = text_tokenize(chunk_data["text"])
            count_terms = Counter(tokens)

            self.list_term_freq.append(count_terms)
            self.doc_length.append(len(tokens))

            doc_total_length += len(tokens)

            for term in count_terms:
                # this is for idf so we can give more weight to rare terms
                self.doc_freq[term] += 1

        self.avg_doc_length = doc_total_length / len(self.chunks)

    def calculate_idf(self, term):
        count_doc = len(self.chunks)

        # common terms will have low score as less informative
        # high importance to rare terms
        count_term_doc = self.doc_freq.get(term, 0)

        if count_term_doc == 0:
            return 0

        numer = count_doc - count_term_doc + 0.5
        denom = count_term_doc + 0.5

        return math.log(numer / denom + 1)

    def score_calculation(self, query_tokens, document_index):
        score_bm25 = 0

        term_count = self.list_term_freq[document_index]
        document_length = self.doc_length[document_index]

        for term in query_tokens:

            # finds bm25 score between one query and one chunk
            if term not in term_count:
                continue

            term_frequency = term_count[term]
            idf = self.calculate_idf(term)

            # term freq of bm25 formula
            numerator = term_frequency * (K1 + 1)

            # this will normalize score so longer docs arent ranked higher
            denominator = term_frequency + K1 * (
                1 - B + B * (document_length / self.avg_doc_length)
            )

            score_bm25 += idf * (numerator / denominator)

        return score_bm25

    def search(self, query, k=10):

        query_tokens = text_tokenize(query)

        # finds bm25 score for each chunk and ranks high to low
        scores_docs = []

        for index in range(len(self.chunks)):

            score = self.score_calculation(query_tokens, index)

            scores_docs.append((index, score))

        scores_docs.sort(key=lambda item: item[1], reverse=True)

        results = []

        for index, score in scores_docs[:k]:

            chunk = self.chunks[index]

            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "score": float(score),
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                }
            )

        return results

    def retrieve(self, query, k=10):
        return self.search(query, k)

    def save_index(self):
        # save only the BM25 data, not the whole class

        index_data = {
            "chunks": self.chunks,
            "list_term_freq": self.list_term_freq,
            "doc_freq": self.doc_freq,
            "doc_length": self.doc_length,
            "avg_doc_length": self.avg_doc_length,
        }

        with open(BM25_INDEX_PATH, "wb") as file:
            pickle.dump(index_data, file)

    @classmethod
    def load_index(cls):

        # this will load the previous generated bm25 index

        with open(BM25_INDEX_PATH, "rb") as file:
            index_data = pickle.load(file)

        retriever = cls.__new__(cls)

        retriever.chunks = index_data["chunks"]
        retriever.list_term_freq = index_data["list_term_freq"]
        retriever.doc_freq = index_data["doc_freq"]
        retriever.doc_length = index_data["doc_length"]
        retriever.avg_doc_length = index_data["avg_doc_length"]

        return retriever


_retriever = None


def bm25_search(query: str, k: int = 3) -> list[dict]:

    global _retriever

    if _retriever is None:

        if BM25_INDEX_PATH.exists():

            _retriever = BM25Retriever.load_index()

        else:

            chunks, _, _ = load_chunks(CHUNKS_PATH)

            _retriever = BM25Retriever(chunks)

            _retriever.save_index()

    return _retriever.retrieve(query, k)


if __name__ == "__main__":

    # check if index is available then reuse it otherwise make new

    if BM25_INDEX_PATH.exists():

        print("Loading existing BM25 index")

        retriever = BM25Retriever.load_index()

    else:

        print("Building BM25 index")

        chunks, _, _ = load_chunks(CHUNKS_PATH)

        print("Loaded chunks:", len(chunks))

        retriever = BM25Retriever(chunks)

        retriever.save_index()

        print("BM25 index saved")

    print("Indexed documents:", len(retriever.chunks))

    print("Average document length:", retriever.avg_doc_length)

    print("Unique terms:", len(retriever.doc_freq))

    final = retriever.retrieve("accordion component")

    for temp in final[:3]:

        print("\nCHUNK:", temp["chunk_id"])
        print("SCORE:", temp["score"])
        print("TEXT:", temp["text"][:200])


def create_bm25_index():
    """Build the BM25 index over the chunk corpus and persist it."""
    chunks, _, _ = load_chunks(CHUNKS_PATH)
    retriever = BM25Retriever(chunks)
    retriever.save_index()
    print(f"bm25 index build complete ({len(chunks)} chunks)")
