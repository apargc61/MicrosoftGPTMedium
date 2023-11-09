import gensim
from gensim import corpora
from pprint import pprint

def preprocess_data(documents):
    # Tokenization, removing stop words, and lemmatization can be added here
    return [[word for word in doc.lower().split()] for doc in documents]

def train_lda(texts):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # You can experiment with the number of topics
    lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    return lda_model, dictionary

def main():
    # Sample customer queries (you can replace these with real data)
    customer_queries = [
        "Tell me about recent stock market updates.",
        "How do I optimize my real estate investments?",
        "Can you provide insights on tax implications for mutual funds?",
        "I'm thinking about retirement funds. Any advice?",
        "How's the gold market doing?"
    ]

    texts = preprocess_data(customer_queries)
    lda_model, dictionary = train_lda(texts)

    # Categorize a new incoming query
    new_query = "What's the latest news on tech stocks?"  # Example new query
    new_query_preprocessed = preprocess_data([new_query])[0]
    new_query_bow = dictionary.doc2bow(new_query_preprocessed)
    topic_distribution = lda_model.get_document_topics(new_query_bow)
    
    # Get the most probable topic
    most_probable_topic = max(topic_distribution, key=lambda x: x[1])
    print(f"The incoming query is most likely about topic {most_probable_topic[0]}")

    # Print topics for inspection
    topics = lda_model.print_topics()
    for topic in topics:
        pprint(topic)

if __name__ == "__main__":
    main()
