from sentence_transformers import SentenceTransformer


def get_sentence_embedding(sentence):
    """
    Get the embedding vector for a sentence.
    
    Args:
        sentence (str): Input sentence
    
    Returns:
        np.ndarray: Sentence embedding vector
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(sentence)
    return embeddings
