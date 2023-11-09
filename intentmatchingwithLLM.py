import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

# Initialize BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to generate embeddings
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    return output['last_hidden_state'][:, 0, :].squeeze().numpy()

# Sample data for intents
intents = {
    "Order pizza": ["I'd like to order a pizza", "Can I get a pizza?", "I want a pizza delivered"],
    "Book a flight": ["I need a flight ticket", "Can I book a flight to Paris?", "I'd like to fly to New York"],
    "Check weather": ["What's the weather like?", "Is it raining today?", "Tell me the weather forecast"]
}

# Generate embeddings for each sample sentence
intent_embeddings = {}
for intent, samples in intents.items():
    intent_embeddings[intent] = [get_embedding(sample) for sample in samples]

# Function to recognize intent
def recognize_intent(query):
    query_embedding = get_embedding(query)
    best_intent = None
    best_similarity = float('-inf')

    for intent, embeddings in intent_embeddings.items():
        for sample_embedding in embeddings:
            similarity = 1 - cosine(query_embedding, sample_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_intent = intent

    if best_similarity > 0.7:  # threshold
        return best_intent
    else:
        return "Unknown intent"

# Test
query = "Is it going to rain tomorrow?"
print(recognize_intent(query))
