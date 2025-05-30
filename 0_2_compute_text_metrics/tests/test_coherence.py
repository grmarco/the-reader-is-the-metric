import unittest
from unittest.mock import MagicMock
from sentence_transformers import SentenceTransformer
from metrics.coherence import TextSplitter, LocalCoherenceAnalyzer, GlobalCoherenceAnalyzer, CoherenceAnalysis
import numpy as np

class MockEmbeddingModel:
    def encode(self, text_list, convert_to_tensor=True):
        # Mock embeddings as random vectors for simplicity
        return np.random.rand(len(text_list), 768)

class TestCoherenceMetrics(unittest.TestCase):
    def setUp(self):
        self.embedding_model = MockEmbeddingModel()

    def test_text_splitter(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        sections = TextSplitter.split_into_sections(text, 2)
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0], "Sentence one Sentence two")
        self.assertEqual(sections[1], "Sentence three Sentence four")

    def test_local_coherence_high(self):
        sentences = [
            "The cat sat on the mat.",
            "The cat sat on the mat.",
            "The cat sat on the mat."
        ]
        analyzer = LocalCoherenceAnalyzer(sentences, self.embedding_model)
        coherence = analyzer.measure_local_coherence()
        self.assertGreater(coherence, 0.9, "High coherence text should have a high coherence score.")

    def test_local_coherence_low(self):
        sentences = [
            "The cat sat on the mat.",
            "The dog barked loudly.",
            "The sun is shining."
        ]
        analyzer = LocalCoherenceAnalyzer(sentences, self.embedding_model)
        coherence = analyzer.measure_local_coherence()
        self.assertLess(coherence, 0.3, "Low coherence text should have a low coherence score.")

    def test_global_coherence_high(self):
        sections = [
            "The cat sat on the mat. The cat sat on the mat.",
            "The cat sat on the mat. The cat sat on the mat."
        ]
        analyzer = GlobalCoherenceAnalyzer(sections)
        coherence = analyzer.topic_modeling_global_coherence()
        self.assertGreater(coherence, 0.5, "High coherence sections should have a high coherence score.")

    def test_global_coherence_low(self):
        sections = [
            "The cat sat on the mat. The dog barked loudly.",
            "The sun is shining. The rain is falling."
        ]
        analyzer = GlobalCoherenceAnalyzer(sections)
        coherence = analyzer.topic_modeling_global_coherence()
        self.assertLess(coherence, 0.3, "Low coherence sections should have a low coherence score.")

    def test_combined_coherence(self):
        text = "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat."
        analyzer = CoherenceAnalysis(text, self.embedding_model)
        coherence_metrics = analyzer.compute_combined_coherence()
        self.assertIn("local_coherence", coherence_metrics)
        self.assertIn("global_coherence_topics", coherence_metrics)
        self.assertGreater(coherence_metrics["local_coherence"], 0.9, "High local coherence expected.")
        self.assertGreater(coherence_metrics["global_coherence_topics"], 0.5, "High global coherence expected.")

if __name__ == '__main__':
    unittest.main()