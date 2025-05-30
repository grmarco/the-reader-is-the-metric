import nltk
from transformers import pipeline
from sentiment import SentimentAnalysis

def test_sentiment_analysis():
    # Inicializar los pipelines de análisis de sentimiento y emoción
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="siebert/sentiment-roberta-large-english",
        device="cuda"
    )
    emotion_pipeline = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,
        device="cuda"
    )
    # Textos representativos con diferentes tonos y emociones
    texts = [
        # Texto positivo largo
        (
            "I am so happy today! Everything is going great and I feel fantastic. "
            "The weather is beautiful, and I had a wonderful breakfast with my family. "
            "We laughed and talked about our plans for the weekend. "
            "I am looking forward to spending time with my friends and enjoying the sunshine. "
            "Life is good and I am grateful for all the positive things happening around me."
        ),
        # Texto negativo largo
        (
            "I am feeling really sad and depressed. Nothing seems to be going right. "
            "I lost my job and I am struggling to find a new one. "
            "My relationships are falling apart, and I feel so alone. "
            "Every day feels like a battle, and I don't know how much longer I can keep going. "
            "I just want to give up and escape from all this pain."
        ),
        # Texto neutral largo
        (
            "Today is an average day. Nothing special happened, just the usual routine. "
            "I woke up, had breakfast, and went to work. "
            "The day passed by without any significant events. "
            "I came back home, had dinner, and watched some TV. "
            "Now I am getting ready for bed, looking forward to another ordinary day tomorrow."
        ),
        # Texto mixto largo
        (
            "I am excited about the new project, but also a bit anxious about the deadlines. "
            "The team is great, and we have some fantastic ideas. "
            "However, the pressure to deliver on time is immense. "
            "I am confident that we can achieve our goals, but the uncertainty is always there. "
            "Balancing excitement and anxiety is challenging, but I am determined to make it work."
        ),
        # Texto con emociones variadas largo
        (
            "The movie was a rollercoaster of emotions. I laughed, cried, and felt scared all at once. "
            "The storyline was captivating, and the characters were so relatable. "
            "There were moments of joy and moments of sorrow, all intertwined beautifully. "
            "The suspense kept me on the edge of my seat, and the ending was both surprising and satisfying. "
            "It was an emotional journey that I will remember for a long time."
        )
    ]
    
    # Probar cada texto
    for i, text in enumerate(texts, start=1):
        analysis = SentimentAnalysis(
            text=text,
            sentiment_pipeline=sentiment_pipeline,
            emotion_pipeline=emotion_pipeline
        )
        results = analysis.compute_sentiment_analysis()
        
        print(f"=== Results for text {i} ===")
        for key, val in results.items():
            print(f"{key}: {val}")
        print()

if __name__ == "__main__":
    test_sentiment_analysis()