import spacy
from sentence_transformers import SentenceTransformer
from coherence import CoherenceAnalysis

def test_coherence_analysis_varied_texts():
    # Load English models
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    spacy_model = spacy.load('en_core_web_sm')
    
    # Multiple English texts with different levels of coherence
    texts = [
        # Relatively coherent text about a single topic
        (
            "London is famous for its history. "
            "Its museums attract millions of tourists every year. "
            "Many visitors enjoy traditional British cuisine. "
            "One can find fish and chips in numerous restaurants."
        ),
        # Text that jumps between topics with no clear cohesion
        (
            "The sun is shining and I like to eat pizza. "
            "Dogs often chase balls in the park. "
            "The universe is expanding faster than we can imagine. "
            "Tomorrow it might rain in Tokyo."
        ),
        # Text with some thread of continuity but mixed concepts
        (
            "The book I read yesterday was about time travel. "
            "A brilliant scientist created the time machine. "
            "However, the main character devoted time to meditation and yoga. "
            "Finally, they discovered that mental health practices helped solve cosmic mysteries."
        ),
    ]
    
    # Test each text
    for i, sample_text in enumerate(texts, start=1):
        analysis = CoherenceAnalysis(
            text=sample_text,
            embedding_model=embedding_model,
            spacy_model=spacy_model,
            words_per_section=2
        )
        results = analysis.compute_coherence(
            num_topics=None,
            start=2,
            limit=6,
            step=1,
            passes=8,
            coherence_type='c_v'
        )
        
        print(f"=== Results for text {i} (English) ===")
        for key, val in results.items():
            print(f"{key}: {val}")
        print()

if __name__ == "__main__":
    test_coherence_analysis_varied_texts()