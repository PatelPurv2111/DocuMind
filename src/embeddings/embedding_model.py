from sentence_transformers import SentenceTransformer
class EmbeddingModel:
    def __init__(self,model_name):
        self.model=SentenceTransformer(model_name)
    def encode(self,texts):
        embeddings=self.model.encode(texts)
        return embeddings
