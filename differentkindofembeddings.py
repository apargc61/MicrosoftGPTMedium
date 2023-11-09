import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import gensim
import gensim.downloader as api

# Sample documents (you can replace these with your own documents)
documents = [
    "Machine learning is a subfield of artificial intelligence.",
    "Python is a popular programming language for data science.",
    "Deep learning models are used for image recognition.",
    "Natural language processing is the field of study for language understanding.",
]

# User query
query = "What is machine learning?"

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# LSA (Latent Semantic Analysis)
lsa = TruncatedSVD(n_components=2)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

# Word2Vec
word2vec_model = gensim.downloader.load("word2vec-google-news-300")
word2vec_matrix = np.array([word2vec_model[word] if word in word2vec_model else np.zeros(300) for word in tfidf_vectorizer.get_feature_names_out()])

# GloVe
glove_model = gensim.downloader.load("glove-wiki-gigaword-300")
glove_matrix = np.array([glove_model[word] if word in glove_model else np.zeros(300) for word in tfidf_vectorizer.get_feature_names_out()])

# spaCy Pretrained Model (en_core_web_md)
nlp = spacy.load("en_core_web_md")
spacy_matrix = np.array([nlp(word).vector for word in tfidf_vectorizer.get_feature_names_out()])

# Calculate similarity between query and documents using cosine similarity
def cosine_similarity(vector1, vector2):
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 == 0 or norm2 == 0:
        return 0  # Avoid division by zero
    return np.dot(vector1, vector2) / (norm1 * norm2)

query_vector = tfidf_vectorizer.transform([query]).toarray().flatten()

embeddings = [
    ("TF-IDF", tfidf_matrix),
    ("LSA", lsa_matrix),
    ("Word2Vec", word2vec_matrix),
    ("GloVe", glove_matrix),
    ("spaCy", spacy_matrix),
]

for embedding_name, embedding_matrix in embeddings:
    similarities = [cosine_similarity(query_vector, doc_vector) for doc_vector in embedding_matrix]
    ranked_indices = np.argsort(similarities)[::-1]  # Sort in descending order
    ranked_documents = [documents[i] for i in ranked_indices]
    
    print(f"{embedding_name} Semantic Search Results:")
    for i, doc in enumerate(ranked_documents, 1):
        print(f"Document {i}: {doc}")
    print()
