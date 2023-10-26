from sentence_transformers import SentenceTransformer, util
import numpy as np

# Sample collection of documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A brown cat is sleeping on the windowsill.",
    "Birds are chirping in the trees.",
    "The dog barks loudly in the park.",
]

# User query
query = "What animals are in the park?"

# Load a pre-trained model (e.g., "bert-base-nli-mean-tokens")
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Encode the documents and the query
document_embeddings = model.encode(documents, convert_to_tensor=True)
print(document_embeddings)
query_embedding = model.encode(query, convert_to_tensor=True)
print(query_embedding)
# Calculate cosine similarity scores
cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)

# Convert cosine similarity scores to a numpy array
cosine_scores = cosine_scores.cpu().numpy()

# Sort documents by similarity score
sorted_documents = [doc for _, doc in sorted(zip(cosine_scores, documents), reverse=True)]

# Print the sorted documents
for i, doc in enumerate(sorted_documents, start=1):
    print(f"Rank {i}: {doc}")
