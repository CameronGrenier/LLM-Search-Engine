<<<<<<< HEAD
"""BM25 sparse retrieval over the chunk index.

Stub. Implement bm25_search to return the same result shape as
dense_search so the generation layer can switch retrievers without
any downstream changes.
"""
from __future__ import annotations


def bm25_search(query: str, k: int = 3) -> list[dict]:
  """Retrieves the top-k chunks for a query by BM25 score.

  Must return the same shape as src.retrieval.dense.dense_search so
  llm.py can treat the two interchangeably:

      [
        {
          "chunk_id": str,
          "text": str,
          "score": float,
          "metadata": dict,   # must include doc_type, url, demo_refs
        },
        ...
      ]

  Args:
    query: Raw user query text.
    k: Number of chunks to return.

  Returns:
    Ranked list of result dicts, highest score first.

  Raises:
    NotImplementedError: Until implemented.
  """
  raise NotImplementedError(
    "bm25_search is not implemented yet. See src/retrieval/dense.py "
    "for the expected result shape."
  )
=======
"""BM25 lexical retrieval."""

from __future__ import annotations
import math
import pickle
import re
from collections import Counter
from config import CHUNKS_PATH, BM25_INDEX_PATH
from src.corpus_io import load_chunks
from src.retrieval.base import Retriever

K1 = 1.5
B = 0.75
#k1 controls how much term freq will affect score
#B controls doc length normalization factor

def text_tokenize(text: str) -> list[str]:
    #this converts text to lowercase so no matching issues
    return re.findall(r"\b\w+\b", text.lower())

class BM25Retriever(Retriever):
    def __init__(self, data_chunks):
        self.chunks = data_chunks
        self.list_term_freq = []
        self.doc_freq = Counter()
        self.doc_length = []
        self.avg_doc_length = 0
        self.index_create()

    def index_create(self):
        #this will make inverted index that bm25 will use
        #each chunk is treated as a document for retrieval
        doc_total_length = 0
        for chunk_data in self.chunks:
            tokens = text_tokenize(chunk_data["text"])
            count_terms = Counter(tokens)
            self.list_term_freq.append(count_terms)
            self.doc_length.append(len(tokens))
            doc_total_length = doc_total_length + len(tokens)
            for term in count_terms:
                #this is for idf so we can give more weight to rare terms
                self.doc_freq[term] = self.doc_freq[term] + 1
        self.avg_doc_length = doc_total_length / len(self.chunks)

    def calculate_idf(self, term):
        count_doc = len(self.chunks)
        #common terms will have low score as less informative
        #high importance to rare terms
        count_term_doc = self.doc_freq.get(
            term,
            0
        )
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
            #finds bm25 score between one query and one chunk
            if term not in term_count:
                continue
            term_frequency = term_count[term]
            idf = self.calculate_idf(term)
            #term freq of bm25 formula
            numerator = term_frequency * (K1 + 1)
            #this will normalize score so longer docs arent ranked higher
            denominator = term_frequency + K1 * (1 - B + B * (document_length / self.avg_doc_length))
            score_bm25 += idf * (numerator / denominator)
        return score_bm25

    def search(self, query, k=10):
        query_tokens = text_tokenize(query)
        #finds bm25 score for each chunk and ranks high to low
        scores_docs = []
        for index in range(len(self.chunks)):
            score = self.score_calculation(
                query_tokens,
                index
            )
            scores_docs.append(
                (index, score)
            )
        scores_docs.sort(
            key=lambda item: item[1],
            reverse=True
        )
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
        #saves built index so we dont need to rebuild bm25 later
        with open(BM25_INDEX_PATH, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_index(cls):
        #this will load the previous generated bm25 index
        with open(BM25_INDEX_PATH, "rb") as file:
            return pickle.load(file)


if __name__ == "__main__":
    #check if index is availab;e then reuse it otherwise make new 
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
    print(
        "Indexed documents:",
        len(retriever.chunks)
    )
    print(
        "Average document length:",
        retriever.avg_doc_length
    )
    print(
        "Unique terms:",
        len(retriever.doc_freq)
    )
    final = retriever.retrieve(
        "accordion component"
    )
    for temp in final[:3]:
        print("\nCHUNK:", temp["chunk_id"])
        print("SCORE:", temp["score"])
        print("TEXT:", temp["text"][:200])
>>>>>>> 567720a (Implement BM25 lexical retrieval)
