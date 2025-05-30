from metrics.thematic import ThematicAnalysis
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# Textos de prueba
text_repetitive = """
The sun rises in the east and sets in the west. The sun brings light and warmth. 
The sun is essential for life. The sun is a star. Plants use sunlight for photosynthesis. 
The sun rises in the east and sets in the west. The sun brings light and warmth.
""" * 2  # Repetimos para aumentar longitud

text_unconnected = """
The ocean is vast and mysterious. Space exploration expands our horizons. 
Ancient civilizations created intricate pyramids. Renewable energy is the future. 
Birds migrate across continents. Artificial intelligence is transforming industries. 
Economic systems evolve over time. Cultural diversity enriches our lives. 
Philosophical ideas shape societies. Music connects people emotionally.
"""

text_interconnected = """
Climate change is a global challenge. It impacts human health, agriculture, and ecosystems. 
Renewable energy technologies, like solar panels and wind turbines, are essential to combat climate change. 
Electric vehicles reduce greenhouse gas emissions. Geopolitical tensions arise from the reliance on rare earth minerals. 
Urban planning integrates green spaces to mitigate heat and improve air quality. 
Artistic movements reflect humanity's response to environmental challenges, fostering awareness and inspiring action.
""" * 2

text_shallow = """
The ocean is mysterious. Artificial intelligence is advancing. 
Ancient civilizations like the Maya are fascinating. Renewable energy sources are crucial for the future. 
Birds migrate across continents. Space exploration is expanding our horizons. 
Music connects people emotionally. Economic systems evolve over time. 
Philosophical ideas shape societies. The sun rises in the east and sets in the west.
""" * 2

# Diccionario de textos
test_texts = {
    "Repetitive": text_repetitive,
    "Unconnected Themes": text_unconnected,
    "Interconnected Themes": text_interconnected,
    "Shallow Exploration": text_shallow,
}

# Resultados y visualización
results = {}

for name, text in test_texts.items():
    print(f"=== Testing: {name} ===\n")
    analyzer = ThematicAnalysis(text)
    thematic_depth_results = analyzer.compute_thematic_depth()
    
    # Guardar resultados
    results[name] = thematic_depth_results

    # Mostrar métricas
    print(f"Thematic Depth Score: {thematic_depth_results['thematic_depth']:.3f}")
    print(f"Entropy Score: {thematic_depth_results['entropy_score']:.3f}")
    print(f"Graph Density: {thematic_depth_results['graph_density']:.3f}")
    print(f"Inter-Theme Similarity: {thematic_depth_results['inter_theme_similarity']:.3f}")
    print("\n")

    # Visualizar grafo
    G = thematic_depth_results['semantic_graph']
    print(f"=== Visualizing Graph for: {name} ===")
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10)
    plt.title(f"Semantic Graph for {name}")
    plt.show()

# Validación de métricas
def validate_results(results):
    """
    Realiza validaciones básicas de los resultados para verificar consistencia.
    """
    thematic_depths = [metrics['thematic_depth'] for metrics in results.values()]
    entropy_scores = [metrics['entropy_score'] for metrics in results.values()]
    graph_densities = [metrics['graph_density'] for metrics in results.values()]
    
    print("=== Validation ===")
    print(f"Mean Thematic Depth: {np.mean(thematic_depths):.3f}")
    print(f"Range of Thematic Depth: {min(thematic_depths):.3f} to {max(thematic_depths):.3f}")
    print(f"Mean Entropy: {np.mean(entropy_scores):.3f}")
    print(f"Range of Entropy: {min(entropy_scores):.3f} to {max(entropy_scores):.3f}")
    print(f"Mean Graph Density: {np.mean(graph_densities):.3f}")
    print(f"Range of Graph Density: {min(graph_densities):.3f} to {max(graph_densities):.3f}")

# Visualización del grafo
def visualize_graph(G, name):
    """
    Guarda y muestra el grafo semántico.
    """
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)})
    plt.title(f"Semantic Graph for {name}")
    plt.savefig(f"{name}_semantic_graph.png")  # Guarda el grafo como una imagen
    plt.close()
    print(f"Graph saved as {name}_semantic_graph.png")

# Validar resultados
validate_results(results)
