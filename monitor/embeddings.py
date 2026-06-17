from sentence_transformers import SentenceTransformer

_model = None


def get_embedding_model():
    """Return a shared SentenceTransformer instance, loading it only once.

    The model is loaded from disk the first time this is called. Every
    subsequent call — across the entire lifetime of the running process —
    reuses that same loaded instance instead of reloading from disk.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model