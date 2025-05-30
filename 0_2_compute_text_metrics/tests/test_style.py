from metrics.style import StylisticRichnessAggregator
import matplotlib.pyplot as plt
import os
import numpy as np

# API Key para el detector de dispositivos retóricos
api_key = ""

# Textos de prueba
test_texts = {
    "Rich in Devices": """
        The forest was a symphony of sounds: the chirping birds, the rustling leaves, and the whispering wind. 
        Time stood still, like a frozen river, as the trees danced in the twilight. 
        Is it not ironic that the fireflies illuminated the darkness better than the sun? 
        The moon, a silent guardian, watched over the sleeping earth. It was a paradox, a deafening silence.
    """,
    "Syntactically Complex": """
        Although the rain poured heavily, the children continued to play, their laughter echoing through the storm. 
        In the corner of the garden, where the roses bloomed, a cat lay lazily, oblivious to the chaos around it. 
        Each drop of rain, as if in unison, created a melody against the tin roof, a rhythm both chaotic and soothing.
    """,
    "Semantically Dense": """
        Quantum mechanics challenges our perception of reality, intertwining particles and waves in a dance of probabilities. 
        The duality of light, both particle and wave, reflects the paradoxes inherent in nature's laws. 
        Entropy, a measure of disorder, governs the universe's irreversible march towards chaos.
    """,
    "Simple Narrative": """
        The boy walked to the store. He bought some milk and bread. On his way home, he saw a dog. 
        The dog barked at him. The boy ran home quickly, holding the groceries tightly.
    """,
    "Mixed Style": """
        The sun was a blazing inferno, dominating the sky, while the gentle waves lapped against the shore. 
        Birds sang songs of joy, a harmony of nature. The fisherman cast his net, his eyes scanning the horizon. 
        What secrets does the ocean hide beneath its shimmering surface? The sun dipped lower, painting the world in hues of orange and pink.
    """
}

# Diccionario para almacenar resultados
results = {}

# Iterar por los textos de prueba
for name, text in test_texts.items():
    print(f"=== Analyzing: {name} ===")
    
    # Instanciar el analizador
    stylistic_analyzer = StylisticRichnessAggregator(text, api_key=api_key)
    
    # Calcular la riqueza estilística
    analysis_results = stylistic_analyzer.compute_stylistic_richness()
    results[name] = analysis_results

    # Imprimir los resultados
    print(f"Stylistic Richness (Aggregated): {analysis_results['stylistic_richness (aggregated)']:.3f}")
    print(f"Linguistic Score (Aggregated): {analysis_results['linguistic_score (aggregated)']:.3f}")
    print(f"Rhetorical Score (Aggregated): {analysis_results['rhetorical_score (aggregated)']:.3f}")
    print(f"Semantic Score (Aggregated): {analysis_results['semantic_score (aggregated)']:.3f}\n")
    
    # Imprimir métricas individuales
    print("Individual Metrics:")
    print(f"Linguistic Variety: {analysis_results['linguistic_variety']}")
    print(f"Rhetorical Devices: {analysis_results['rhetorical_devices']}")
    print(f"Semantic Density: {analysis_results['semantic_density']}")
    print("\n" + "="*50 + "\n")

# Graficar Resultados
def plot_results(results, output_dir="figures"):
    """
    Genera gráficos comparativos de los puntajes agregados y guarda gráficos individuales de métricas.
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Preparar datos para el gráfico
    labels = results.keys()
    stylistic_scores = [result["stylistic_richness (aggregated)"] for result in results.values()]
    linguistic_scores = [result["linguistic_score (aggregated)"] for result in results.values()]
    rhetorical_scores = [result["rhetorical_score (aggregated)"] for result in results.values()]
    semantic_scores = [result["semantic_score (aggregated)"] for result in results.values()]

    x = np.arange(len(labels))

    # Crear gráfico principal
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, stylistic_scores, 0.2, label="Stylistic Richness (Aggregated)")
    plt.bar(x, linguistic_scores, 0.2, label="Linguistic Score (Aggregated)")
    plt.bar(x + 0.2, rhetorical_scores, 0.2, label="Rhetorical Score (Aggregated)")
    plt.bar(x + 0.4, semantic_scores, 0.2, label="Semantic Score (Aggregated)")

    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Score")
    plt.title("Stylistic Richness and Component Scores (Aggregated)")
    plt.legend()
    plt.tight_layout()

    # Guardar gráfico principal
    main_output_path = os.path.join(output_dir, "stylistic_richness_scores.png")
    plt.savefig(main_output_path)
    plt.close()
    print(f"Main graph saved in: {main_output_path}")

    # Graficar métricas individuales por texto
    for name, result in results.items():
        individual_metrics = result["linguistic_variety"]
        plt.figure(figsize=(10, 5))
        plt.bar(individual_metrics.keys(), individual_metrics.values(), color='skyblue')
        plt.title(f"Linguistic Variety for {name}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        individual_output_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_linguistic_metrics.png")
        plt.savefig(individual_output_path)
        plt.close()
        print(f"Individual graph for {name} saved in: {individual_output_path}")

# Ejecutar función para graficar y guardar
plot_results(results)
