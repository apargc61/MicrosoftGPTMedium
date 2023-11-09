from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "Machine learning is a subfield of artificial intelligence.",
    "Python is a popular programming language for data science.",
    "Deep learning models are used for image recognition.",
    "Natural language processing is the field of study for language understanding.",
]

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents into TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print(tfidf_matrix)
# tfidf_matrix now contains the TF-IDF vectors for the documents
