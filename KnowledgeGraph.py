import spacy
import networkx as nx

# Load a spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Example text data (you would typically load a large dataset)
text_data = [
    "Apple Inc. is headquartered in Cupertino, California.",
    "Steve Jobs was a co-founder of Apple Inc.",
    "The iPhone is a popular product by Apple.",
    "Tesla, Inc. is known for electric cars and renewable energy solutions.",
    "Elon Musk is the CEO of Tesla, Inc."
]

# Initialize a Knowledge Graph using NetworkX
knowledge_graph = nx.Graph()

# Process each text and extract entities and relationships
for text in text_data:
    doc = nlp(text)
    print("doc", doc)
    # Extract entities (e.g., organizations, persons)
    entities = [ent.text for ent in doc.ents]
    print("entitits", entities)
    # Create nodes for entities if they don't exist in the graph
    for entity in entities:
        if not knowledge_graph.has_node(entity):
            knowledge_graph.add_node(entity)
    
    # Extract relationships (e.g., "is headquartered in")
    relationships = [(token.head.text, token.text) for token in doc if token.dep_ == "prep"]
    print("relationships", relationships)
    # Create edges for relationships in the graph
    for relationship in relationships:
        knowledge_graph.add_edge(relationship[0], relationship[1])

# Visualize the knowledge graph (optional)
import matplotlib.pyplot as plt
pos = nx.spring_layout(knowledge_graph)
nx.draw(knowledge_graph, pos, with_labels=True, node_size=800, node_color="skyblue")
plt.show()

# Query the knowledge graph
print("Entities:", knowledge_graph.nodes())
print("Relationships:", knowledge_graph.edges())
