import numpy as np
from sentence_transformers import SentenceTransformer
from thematic import ThematicAnalysis

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar un modelo de SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Puedes elegir otro modelo si lo prefieres

    texts = {
    "historia_ciencia": """
    Durante la Edad Media, los reinos de Europa experimentaron grandes cambios políticos y sociales. 
    Los sistemas feudales dominaban la estructura social, con los señores feudales controlando vastas 
    extensiones de tierra mientras los campesinos trabajaban para ellos. Sin embargo, la llegada de 
    la ciencia moderna trajo consigo una transformación del pensamiento. Galileo Galilei y Copérnico 
    desafiaron las creencias tradicionales con sus observaciones astronómicas, poniendo las bases 
    para la Revolución Científica y el método empírico.
    """,
    
    "filosofia_literatura": """
    Platón sostenía que el mundo que percibimos es solo una sombra de la verdadera realidad, la de las 
    formas ideales. Mientras tanto, en la literatura española, Miguel de Cervantes exploraba la dualidad 
    entre realidad y ficción a través de su obra maestra, 'Don Quijote de la Mancha'. El protagonista, 
    perdido entre la fantasía de los libros de caballería, desafía la percepción común de la realidad, 
    reflejando un dilema filosófico sobre la naturaleza de la verdad.
    """,
    
    "tecnologia_musica": """
    El desarrollo de la inteligencia artificial ha transformado la industria musical. Algoritmos de 
    aprendizaje profundo son capaces de generar melodías y ritmos, mientras que herramientas de 
    procesamiento de audio permiten la manipulación avanzada del sonido. Sin embargo, los compositores 
    clásicos como Beethoven y Mozart dependían de la armonía y la estructura formal, explorando la belleza 
    a través de la simplicidad matemática de las progresiones tonales. La tecnología actual permite 
    reinterpretar estas obras con nuevos enfoques sonoros.
    """,
    
    "deporte_naturaleza": """
    El senderismo es una de las actividades más saludables y enriquecedoras para el cuerpo y la mente. 
    Caminar por senderos naturales no solo mejora la salud cardiovascular, sino que también reduce el 
    estrés y conecta a las personas con la naturaleza. Mientras tanto, deportes de alta intensidad como 
    el fútbol y el baloncesto requieren un nivel de resistencia y coordinación mucho mayor, enfocados 
    en la competencia y el rendimiento físico de los atletas profesionales.
    """,

    "politica_ciencia": """
    Los debates sobre el cambio climático han alcanzado el ámbito político global. Las decisiones sobre 
    emisiones de carbono y la transición a energías renovables afectan directamente a las políticas 
    nacionales e internacionales. Mientras tanto, la investigación científica sobre el calentamiento 
    global, liderada por organizaciones como el IPCC, ofrece datos empíricos sobre la relación entre 
    la actividad humana y el aumento de las temperaturas globales, alimentando un debate entre evidencia 
    científica y decisiones políticas.
    """
}

    texts["mezcla_extrema"] = """
    La fotosíntesis es un proceso bioquímico crucial para las plantas. Mientras tanto, en la Revolución Francesa, 
    los ideales de libertad y fraternidad marcaron un cambio radical en la historia europea. Los agujeros negros, 
    regiones del espacio-tiempo con gravedad extrema, desafían nuestra comprensión de la física clásica. 
    El arte del Renacimiento italiano, con obras de Leonardo da Vinci y Miguel Ángel, buscaba capturar la belleza 
    y la perspectiva de manera realista. Paralelamente, los algoritmos de machine learning permiten la clasificación 
    automática de imágenes y textos. En la Edad Media, los feudos organizaban la producción agrícola mientras se 
    desarrollaban nuevas teorías filosóficas sobre el alma y la existencia. 
"""
    texts["tema_unico"] = """
    La fotosíntesis es el proceso mediante el cual las plantas convierten la luz solar en energía química. 
    Este proceso ocurre en los cloroplastos, orgánulos celulares que contienen clorofila. La luz es absorbida 
    por la clorofila, lo que permite la conversión de dióxido de carbono y agua en glucosa y oxígeno. 
    La glucosa es utilizada como fuente de energía para el crecimiento de la planta, mientras que el oxígeno 
    se libera a la atmósfera. Este mecanismo es fundamental para la vida en la Tierra, ya que mantiene el equilibrio 
    de oxígeno y carbono en la atmósfera.
"""
    texts["cambio_gradual"] = """
    La historia de la astronomía comienza con las primeras civilizaciones que observaron los cielos para 
    predecir eventos estacionales. Los babilonios y egipcios desarrollaron calendarios basados en la posición 
    de los astros. Con la llegada de la Revolución Científica, Copérnico propuso un modelo heliocéntrico del 
    sistema solar, revolucionando nuestra comprensión del universo. Esta idea fue luego refinada por Kepler, 
    cuyas leyes del movimiento planetario describieron órbitas elípticas. En la física moderna, Einstein llevó 
    la exploración del cosmos más allá, introduciendo la teoría de la relatividad general y prediciendo fenómenos 
    como los agujeros negros.
"""
    texts["temas_separados"] = """
    La evolución de los dinosaurios marcó una era de dominación en el período Mesozoico. Estos reptiles 
    habitaron la Tierra durante millones de años, diversificándose en formas como el Tiranosaurio Rex y 
    el Triceratops. Se cree que un evento de extinción masiva, posiblemente un impacto de asteroide, 
    acabó con su reinado. 

    Ahora cambiamos de tema completamente: La música barroca, representada por compositores como Bach y 
    Vivaldi, se caracteriza por la complejidad de sus estructuras armónicas. Las fugas y conciertos de 
    Bach muestran un dominio excepcional de la contrapuntística.
"""
    texts["circular"] = """
    La inteligencia artificial ha revolucionado la medicina, permitiendo diagnósticos asistidos por 
    algoritmos. Por ejemplo, en el análisis de imágenes médicas, los modelos de deep learning han alcanzado 
    precisión comparable a la de radiólogos humanos. 

    En la astronomía, el uso de inteligencia artificial ha permitido la identificación de exoplanetas 
    mediante el análisis de datos de telescopios espaciales. 

    Regresando a la medicina, las herramientas de IA también se usan para predecir la progresión de 
    enfermedades como el cáncer y la diabetes. Estas aplicaciones demuestran el potencial de la IA para 
    transformar múltiples campos del conocimiento.
"""

    # Inicializar el modelo de embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Probar distintas configuraciones de segmentación
    segment_sizes = [10]  # Diferentes tamaños de segmentos para comparar resultados

    # Almacenar resultados para comparar
    results = {}

    # Iterar sobre cada texto y cada tamaño de segmentación
    for text_name, text_content in texts.items():
        print(f"\n\nEvaluando texto: {text_name}")
        results[text_name] = {}
        
        for segment_size in segment_sizes:
            print(f"\nSegmentación con {segment_size} palabras por segmento:")
            
            # Inicializar el análisis temático con el tamaño de segmento especificado
            analysis = ThematicAnalysis(
                text=text_content,
                embedding_model=embedding_model,
                words_per_segment=segment_size,
                clustering_threshold=0.55,
                graph_similarity_threshold=0.2
            )
            
            # Calcular métricas
            metrics = analysis.compute_thematic_depth(visualize=True)
            results[text_name][segment_size] = metrics
            
            # Imprimir resultados para el texto y segmentación actual
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

    # Comparación general de resultados
    print("\n\nComparación de Resultados:")
    for text_name, segment_data in results.items():
        print(f"\nTexto: {text_name}")
        for segment_size, metrics in segment_data.items():
            print(f" - Segmentación de {segment_size} palabras:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")

