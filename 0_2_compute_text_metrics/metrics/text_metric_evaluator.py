import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import spacy
import numpy as np

from style import StylisticRichnessAggregator
from coherence import CoherenceAnalysis
from originality import OriginalityAnalysis
from sentiment import SentimentAnalysis
from thematic import ThematicAnalysis
from readability import ReadabilityAnalysis

api_key = ""

class TextMetricEvaluator:
    def __init__(self, embedding_model, lm_tokenizer, lm_model, spacy_nlp, sentiment_pipeline, emotion_pipeline):
        self.embedding_model = embedding_model
        self.lm_tokenizer = lm_tokenizer
        self.lm_model = lm_model
        self.spacy_nlp = spacy_nlp
        self.sentiment_pipeline = sentiment_pipeline
        self.emotion_pipeline = emotion_pipeline

    def compute_metrics(self, text, reference_texts):
        words_per_segment = 15
        coherence_analysis = CoherenceAnalysis(text=text, embedding_model=self.embedding_model, spacy_model=self.spacy_nlp, words_per_section=words_per_segment)
        originality_analysis = OriginalityAnalysis(
            text,
            reference_texts,
            self.lm_model,
            self.lm_tokenizer,
            self.embedding_model
        )
        stylistic_analysis = StylisticRichnessAggregator(
            text=text,
            embedding_model=self.embedding_model,
            spacy_model=self.spacy_nlp,
            api_key=api_key
        )
        sentiment_analysis = SentimentAnalysis(
            text,
            self.sentiment_pipeline,
            self.emotion_pipeline,
            chunk_size=words_per_segment
        )
        thematic_analysis = ThematicAnalysis(
                    text=text,
                    embedding_model=self.embedding_model,
                    spacy_model=self.spacy_nlp,
                    words_per_segment=words_per_segment,
                    use_lemmatization=True,  
                    clustering_threshold=0.6,  
                    graph_similarity_threshold=0.45,  
                    linkage='average',
                )

        readability_metrics = ReadabilityAnalysis(language='en')

        coherence_metrics = coherence_analysis.compute_coherence(num_topics=None)
        readability_metrics = readability_metrics.analyze(text)
        originality_metrics = originality_analysis.compute_originality_score()
        stylistic_metrics = stylistic_analysis.compute_stylistic_richness()
        sentiment_metrics = sentiment_analysis.compute_sentiment_analysis()
        thematic_metrics = thematic_analysis.compute_thematic_depth()

        detailed_report = {
            "coherence_analysis": coherence_metrics,
            "originality_analysis": originality_metrics,
            "stylistic_analysis": stylistic_metrics,
            "sentiment_analysis": sentiment_metrics,
            "thematic_analysis": thematic_metrics,
            "readability_metrics": readability_metrics
        }
        return detailed_report

    def save_metrics_to_json(self, metrics, output_file):
        def convert_to_serializable(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)

        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4, default=convert_to_serializable)