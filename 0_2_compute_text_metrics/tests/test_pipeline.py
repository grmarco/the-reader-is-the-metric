from src.metrics.compute_metrics import TextAnalysisPipeline

# Textos de prueba
texts = {
    "Nature's Symphony": """
        The forest was alive with the sounds of chirping birds and rustling leaves. Sunlight filtered 
        through the dense canopy, creating a mosaic of light and shadow on the forest floor. A gentle 
        breeze carried the sweet scent of blooming flowers, mingling with the earthy aroma of damp soil. 
        As the sun set, painting the sky in hues of orange and pink, a sudden storm disrupted the peace, 
        sending animals scurrying for shelter. Yet, even amidst the chaos, the forest retained its beauty, 
        a breathtaking rainbow arching over the treetops as the storm passed. The balance of nature was 
        restored, a testament to its resilience.
    """,
    "The Rise and Fall of Empires": """
        The Roman Empire, one of history's greatest civilizations, rose from humble beginnings to dominate 
        much of the ancient world. Its legacies in law, architecture, and governance continue to influence 
        societies today. However, the seeds of its decline were sown within its grandeur. Political corruption, 
        economic instability, and invasions by barbarian tribes chipped away at its foundations. The empire 
        eventually split into eastern and western halves, with the Western Roman Empire collapsing in 476 AD. 
        Meanwhile, the Eastern Roman Empire, known as the Byzantine Empire, thrived for centuries before falling 
        to the Ottoman Turks in 1453. The story of Rome is a reminder of the cyclical nature of history.
    """,
    "The Digital Age: A Double-Edged Sword": """
        The advent of the internet revolutionized communication, commerce, and culture. Social media platforms 
        have connected people across the globe, fostering relationships and spreading information at unprecedented 
        speeds. Yet, this connectivity comes with risks. Privacy concerns, misinformation, and the rise of 
        cyberbullying are stark reminders of the internet's darker side. The digital age has also disrupted 
        traditional industries, forcing businesses to adapt to rapidly changing technologies. As we navigate 
        this digital landscape, the challenge lies in harnessing its potential for good while mitigating its 
        negative impacts. The future of the digital age depends on the choices we make today.
    """,
    "Adventures Beneath the Waves": """
        The ocean, covering more than 70% of Earth's surface, remains one of the planet's greatest mysteries. 
        Beneath its waves lies a world teeming with life, from colorful coral reefs to the eerie depths of 
        the midnight zone. Marine biologists have discovered species adapted to extreme conditions, such as 
        hydrothermal vents and freezing polar waters. Yet, human activity threatens these delicate ecosystems. 
        Overfishing, pollution, and climate change have pushed many marine species to the brink of extinction. 
        Despite these challenges, efforts to protect the oceans are gaining momentum. Conservation initiatives 
        and sustainable practices offer hope for preserving this underwater wonderland for future generations.
    """,
    "The Human Spirit in Times of Adversity": """
        Throughout history, the human spirit has proven resilient in the face of adversity. During times of 
        war, famine, and natural disasters, communities have come together to rebuild and support one another. 
        Stories of courage and perseverance abound, from the bravery of soldiers on the battlefield to the 
        quiet determination of individuals overcoming personal struggles. In recent times, the COVID-19 pandemic 
        has tested the limits of human endurance, yet it has also revealed our capacity for innovation and 
        compassion. Vaccines were developed at record speed, and acts of kindness brought hope to those in need. 
        The human spirit, unyielding and adaptable, continues to inspire.
    """,
    "Exploring the Cosmos": """
        Humanity's fascination with the stars stretches back to ancient times, when civilizations charted the 
        heavens to navigate the seas and mark the passage of seasons. Today, space exploration is entering a 
        new era, driven by advances in technology and a renewed sense of curiosity. From Mars rovers to the 
        James Webb Space Telescope, scientists are uncovering the secrets of the universe. Private companies 
        are also pushing the boundaries of what's possible, envisioning a future where space travel is accessible 
        to all. Yet, with this progress comes responsibility. Preserving the cosmic environment and ensuring 
        ethical exploration will shape the legacy of humanity's journey to the stars.
    """
}

# Directorio de salida para los informes
output_dir = "reports"

# Iterar por los textos de prueba
for name, text in texts.items():
    print(f"=== Generating Report for: {name} ===")
    
    # Crear instancia de la pipeline
    pipeline = TextAnalysisPipeline(
        text=text,
        reference_texts=list(texts.values()),
        output_dir=f"{output_dir}_{name}",
    )
    
    # Ejecutar el análisis
    pipeline.run_analysis()

    print(f"Report for {name} saved in {output_dir}\n")
