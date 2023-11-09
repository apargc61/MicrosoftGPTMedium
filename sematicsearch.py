import spacy
import numpy as np

# Load spaCy's pre-trained model with word embeddings
nlp = spacy.load("en_core_web_md")

# Sample documents (you can replace these with your own documents)
documents = [
    "Machine learning is a subfield of artificial intelligence.",
    "Python is a popular programming language for data science.",
    "Deep learning models are used for image recognition.",
    "Natural language processing is the field of study for language understanding.",
]

# User query
query = "What is machine learning?"

# Calculate similarity between query and documents
query_vector = nlp(query).vector
print(query_vector)
similarities = []

for doc in documents:
    doc_vector = nlp(doc).vector
    similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
    similarities.append((doc, similarity))

# Sort documents by similarity
similarities.sort(key=lambda x: x[1], reverse=True)

# Display top matching documents
for i, (doc, similarity) in enumerate(similarities, 1):
    print(f"Document {i} ({similarity:.2f}): {doc}")
