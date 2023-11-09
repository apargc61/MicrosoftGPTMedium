import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.tokenize import word_tokenize

# Step 1: Data Collection
# Assuming you have a dataset with 'text' and 'label' columns
data = pd.read_csv('your_dataset.csv')

# Step 2: Data Preprocessing
# Tokenization, lowercasing, and removing stopwords and noise
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(tokens)

data['text'] = data['text'].apply(preprocess_text)
print(data)
# Step 4: Feature Extraction
X = data['text']  # Text data
y = data['label']  # Target labels

# Convert text data to TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features
X_tfidf = tfidf_vectorizer.fit_transform(X)
df = pd.DataFrame.sparse.from_spmatrix(X_tfidf)
print(df)

# Step 5: Model Building
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize and train a classifier (e.g., Naive Bayes)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Step 6: Model Training and Evaluation
# ...

# Step 7: Get Feature Importance
feature_names = tfidf_vectorizer.get_feature_names_out()
print(feature_names)
# Calculate feature log probabilities
feature_log_probabilities = classifier.feature_log_prob_

# Create a DataFrame to display feature importance scores
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_log_probabilities[0]})

# Sort features by importance (descending order)
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print or visualize the top features
print(feature_importance_df)#.head(20))  # Print the top 10 important features
