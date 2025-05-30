# Importa la clase `SentimentAnalysis`
from metrics.sentiment import SentimentAnalysis  # Asegúrate de usar el nombre correcto del módulo y archivo
import matplotlib.pyplot as plt
import os

# Textos de prueba
texts = {
    "Emotional Growth": """
        In the beginning, there was chaos. The world seemed hostile, and every step forward was a battle. 
        Yet, amid the darkness, there was a spark of hope. Slowly, people began to rebuild. 
        Laughter returned to the streets, and the sounds of joy echoed across the towns. 
        The fear of the unknown gave way to excitement for new possibilities.
        By the end, unity and love prevailed, creating a world brighter than ever imagined.
    """,
    "Emotional Turmoil": """
        The party started with laughter and joy. People danced, shared stories, and celebrated life. 
        Suddenly, the lights went out, and panic filled the room. Whispers of fear turned into loud cries. 
        Relief came as the lights returned, but then anger emerged over the disruption.
        A heartfelt apology shifted the atmosphere to calm, followed by forgiveness and renewed cheer.
    """,
    "Mixed Information": """
        The Earth revolves around the sun in 365 days, providing seasons and time for renewal. 
        In a small village, children played in the fields, their laughter mixing with the rustle of the wind.
        The progress of science brought new opportunities, but it also raised ethical concerns.
        Amid this mix, the resilience of communities proved that humanity could thrive despite challenges.
    """,
    "Neutral Reflection": """
        Photosynthesis is the process by which plants convert sunlight into energy. 
        The oceans cover 70% of Earth's surface, playing a vital role in regulating climate.
        The intricate balance of ecosystems ensures the survival of countless species.
        These facts highlight the interconnectedness of life on our planet.
    """
}

def save_emotion_trajectory(emotion_trajectory, title, output_dir="emotion_charts"):
    """
    Guarda la trayectoria emocional del texto en un archivo PNG.
    Ajusta la leyenda para mejorar la visualización.
    """
    # Asegurarse de que el directorio de salida existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear el gráfico
    plt.figure(figsize=(12, 8))  # Ajustar el tamaño del gráfico
    for emotion, values in emotion_trajectory.items():
        if max(values) > 0:  # Solo graficar emociones presentes
            plt.plot(values, label=emotion)

    plt.title(title)
    plt.xlabel("Sentence")
    plt.ylabel("Intensity")
    
    # Colocar la leyenda fuera del gráfico
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize="small")
    plt.grid(True)
    
    # Guardar el gráfico en un archivo
    filename = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(filename, format="png", bbox_inches="tight")  # Ajustar el gráfico al contenido
    plt.close()  # Cerrar la figura para liberar memoria
    print(f"[INFO] Graph saved as: {filename}")

# Itera sobre los textos y ejecuta el análisis
device = 0  # Configura el dispositivo (0 para GPU, -1 para CPU)
for name, text in texts.items():
    print(f"=== Testing: {name} ===")
    
    # Inicializa el análisis
    analyzer = SentimentAnalysis(text, device=device)
    
    # Ejecuta el análisis
    results = analyzer.compute_sentiment_analysis()
    
    # Imprime los resultados
    print(f"Sentiment Strength: {results['sentiment_strength']:.3f}")
    print(f"Emotional Richness: {results['emotional_richness']:.3f}")
    print(f"Temporal Dynamics: {results['temporal_dynamics']:.3f}")
    print(f"Emotional Impact Score: {results['emotional_impact_score']:.3f}")
    
    # Guardar la trayectoria emocional si está disponible
    if "emotion_trajectory" in results:
        save_emotion_trajectory(results["emotion_trajectory"], f"Emotion Trajectory: {name}")
