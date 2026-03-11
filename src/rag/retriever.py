def retrieve(vector_db, query_embedding):

    docs = vector_db.search(query_embedding)

    return docs