import networkx as nx

# Create a sample knowledge graph
knowledge_graph = nx.DiGraph()

# Define nodes and relationships in the knowledge graph
knowledge_graph.add_node("Customer", label="Person")
knowledge_graph.add_node("Account", label="Financial Product")
knowledge_graph.add_edge("Customer", "Account", relationship="HAS_ACCOUNT")

# Sample data in the knowledge graph
knowledge_graph.nodes["Customer"]["data"] = {"name": "John Doe", "age": 30}
knowledge_graph.nodes["Account"]["data"] = {"account_number": "12345", "balance": 5000}
print(knowledge_graph)
# User query
user_query = "Tell me about John Doe's account."

# Tokenize the user query (you may use an NLP library for more advanced parsing)
tokens = user_query.lower().split()

# Extract entities from the user query
entities = [token for token in tokens if token in knowledge_graph.nodes]

# Perform a simple graph query to extract information
if entities:
    entity = entities[0]  # Assuming one entity per query
    entity_data = knowledge_graph.nodes[entity]["data"]
    print(f"Information about {entity}: {entity_data}")
else:
    print("Sorry, I couldn't find any information.")
