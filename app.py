from src.config.settings import settings

from src.data.document_loader import load_documents
from src.data.cleaner import clean_text
from src.data.chunker import chunk_text

from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.faiss_db import VectorDB

from src.llm.llm_model import LLM
from src.rag.rag_pipeline import run_rag


def main():

    print("Loading documents...")

    docs = load_documents(settings.RAW_DATA_PATH)

    cleaned_docs = [clean_text(doc) for doc in docs]

    print("Chunking documents...")

    all_chunks = []

    for doc in cleaned_docs:

        chunks = chunk_text(doc)

        all_chunks.extend(chunks)

    print("Generating embeddings...")

    embedder = EmbeddingModel(settings.EMBEDDING_MODEL)

    embeddings = embedder.encode(all_chunks)

    print("Building vector database...")

    vector_db = VectorDB(len(embeddings[0]))

    vector_db.add(embeddings, all_chunks)

    llm = LLM()

    print("\nDocuMind Ready\n")

    while True:

        query = input("Ask Question > ")

        if query.lower() == "exit":
            break

        answer = run_rag(query, embedder, vector_db, llm)

        print("\nAnswer:\n", answer)
        print("\n")


if __name__ == "__main__":

    main()