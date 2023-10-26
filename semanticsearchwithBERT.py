from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample collection of documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A brown cat is sleeping on the windowsill.",
    "Birds are chirping in the trees.",
    "The dog barks loudly in the park.",
]

# User query
query = "What animals are in the park?"

# Load BERT-based model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode the query and documents
query_tokens = tokenizer.encode(query, add_special_tokens=True, return_tensors='pt')
document_tokens = [tokenizer.encode(doc, add_special_tokens=True, return_tensors='pt') for doc in documents]

# Calculate embeddings for query and documents
with torch.no_grad():
    query_embedding = model(query_tokens)[0].mean(dim=1)  # Mean pooling of token embeddings
    print(query_embedding)
    document_embeddings = [model(doc_token)[0].mean(dim=1) for doc_token in document_tokens]
    print(document_embeddings)

# Calculate cosine similarity scores
scores = [cosine_similarity(query_embedding.numpy(), doc_embedding.numpy())[0][0] for doc_embedding in document_embeddings]

# Sort documents by similarity score
sorted_documents = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]

# Print the sorted documents
for i, doc in enumerate(sorted_documents, start=1):
    print(f"Rank {i}: {doc}")


# from transformers import BertTokenizer, BertModel
# import torch
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# # Sample collection of documents
# documents = [
#     "The quick brown fox jumps over the lazy dog.",
#     "A brown cat is sleeping on the windowsill.",
#     "Birds are chirping in the trees.",
#     "The dog barks loudly in the park.",
# ]

# # User query
# query = "What animals are in the park?"

# # Load BERT-based model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# # Encode the query and documents
# query_tokens = tokenizer.encode(query, add_special_tokens=True)
# document_tokens = [tokenizer.encode(doc, add_special_tokens=True) for doc in documents]

# # Calculate embeddings for query and documents
# with torch.no_grad():
#     query_embedding = model(torch.tensor([query_tokens]))[0][0]  # Extract CLS token representation
#     document_embeddings = [model(torch.tensor([doc]))[0][0] for doc in document_tokens]

# # Ensure that embeddings have the same dimensions
# embedding_dim = query_embedding.shape[0]
# document_embeddings = [doc_embedding[:embedding_dim] for doc_embedding in document_embeddings]

# # Calculate cosine similarity scores
# scores = [cosine_similarity([query_embedding], [doc_embedding])[0][0] for doc_embedding in document_embeddings]

# # Sort documents by similarity score
# sorted_documents = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]

# # Print the sorted documents
# for i, doc in enumerate(sorted_documents, start=1):
#     print(f"Rank {i}: {doc}")
