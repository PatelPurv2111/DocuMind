import gradio as gr

from src.config.settings import settings
from src.data.document_loader import load_documents
from src.data.cleaner import clean_text
from src.data.chunker import chunk_text
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.faiss_db import VectorDB
from src.llm.llm_model import LLM
from src.rag.rag_pipeline import run_rag


print("Loading documents...")

docs = load_documents(settings.RAW_DATA_PATH)

cleaned_docs = [clean_text(doc) for doc in docs]

all_chunks = []

for doc in cleaned_docs:
    chunks = chunk_text(doc)
    all_chunks.extend(chunks)

print("Generating embeddings...")

embedder = EmbeddingModel(settings.EMBEDDING_MODEL)

embeddings = embedder.encode(all_chunks)

vector_db = VectorDB(len(embeddings[0]))

vector_db.add(embeddings, all_chunks)

llm = LLM()

print("DocuMind Ready")


def ask_question(question):

    answer = run_rag(question, embedder, vector_db, llm)

    return answer


interface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask something about the documents..."),
    outputs="text",
    title="DocuMind - Document Intelligence System",
    description="Ask questions from the document knowledge base using RAG."
)

interface.launch()