import gensim
from gensim import corpora
from pprint import pprint

# Sample documents
documents = [
    "Sugar is bad to consume. My sister likes to have sugar, but not my father.",
    "My father spends a lot of time driving my sister around to dance practice.",
    "Doctors suggest that driving may cause increased stress and blood pressure.",
    "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better.",
    "Health experts say that sugar is not good for your lifestyle."
]

# Tokenize the documents
texts = [[text for text in doc.split()] for doc in documents]

# Create a dictionary from the tokenized documents
dictionary = corpora.Dictionary(texts)

# Create a corpus from the dictionary
corpus = [dictionary.doc2bow(text) for text in texts]

# LDA Model
num_topics = 3
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# Print the topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)
