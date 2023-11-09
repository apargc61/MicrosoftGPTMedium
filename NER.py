import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California. " \
       "It is known for its iPhones and MacBooks."

# Process the text
doc = nlp(text)
print(doc.ents)
# Extract named entities
entities = [(ent.text, ent.label_) for ent in doc.ents]

# Print the named entities and their labels
for entity, label in entities:
    print(f"Entity: {entity}, Label: {label}")
