from src.fetch_corpus import fetch_corpus
from src.preprocess import preprocess
from src.chunk import chunk
from src.retrieval.dense import create_dense_index
from src.retrieval.bm25 import create_bm25_index

def main() -> None:
  fetch_corpus()
  preprocess()
  chunk()
  create_dense_index()
  create_bm25_index()