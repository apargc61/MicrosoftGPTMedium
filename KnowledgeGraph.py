import networkx as nx
import matplotlib.pyplot as plt

def draw_knowledge_graph():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes (entities)
    G.add_node("Albert Einstein")
    G.add_node("Physics")
    G.add_node("Germany")

    # Add edges (relationships)
    G.add_edge("Albert Einstein", "Physics", relation="worked in field of")
    G.add_edge("Albert Einstein", "Germany", relation="born in")

    # Draw the graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))
    
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=15, font_weight='bold', width=2, edge_color="gray")
    
    # Draw edge labels
    edge_labels = {(u, v): G[u][v]['relation'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

    plt.title("Simple Knowledge Graph")
    plt.show()

draw_knowledge_graph()
