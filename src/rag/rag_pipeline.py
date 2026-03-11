from src.rag.retriever import retrieve


def run_rag(query, embedder, vector_db, llm):

    query_embedding = embedder.encode([query])

    retrieved_docs = retrieve(vector_db, query_embedding)

    context = " ".join(retrieved_docs)

    prompt = f"""
You are an AI assistant.

Answer the question using ONLY the information from the context.

Give a short answer (1 sentences).

Context:
{context}

Question: {query}

Answer:
"""
    answer = llm.generate(prompt)

    return answer