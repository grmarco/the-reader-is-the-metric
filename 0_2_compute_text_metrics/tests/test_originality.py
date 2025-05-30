import unittest
from metrics.originality import OriginalityAnalysis
import numpy as np

class TestOriginalityMetrics(unittest.TestCase):
    def setUp(self):
        # Corpus de referencia
        self.reference_corpus = [
            "Artificial intelligence is transforming industries and creating opportunities.",
            "The Roman Empire fell, marking the beginning of the Middle Ages in European history.",
            "Shakespeare's plays are considered a cornerstone of English literature.",
            "The mysteries of the universe have fascinated scientists and philosophers for centuries.",
            "The vibrant colors of the forest reflected the diversity of life it hosted.",
        ]

        # Textos de prueba
        self.test_texts = {
            "High Originality (Short)": "Quantum computing is an emerging technology with immense potential.",
            "Moderate Originality (Short)": "The Roman Empire fell, ushering in a new era of European history.",
            "Low Originality (Short)": "Artificial intelligence is transforming industries and creating opportunities.",
            "High Originality (Medium)": """
            Beneath the ocean's surface, bioluminescent creatures light up the abyss, creating a spectacle of natural beauty. 
            These organisms have evolved in complete darkness, showcasing the ingenuity of life.
            """,
            "Moderate Originality (Medium)": """
            The Roman Empire left an indelible mark on European culture, influencing art, architecture, and governance. 
            Its legacy continues to inspire modern societies.
            """,
            "Low Originality (Medium)": """
            Artificial intelligence is transforming industries and creating opportunities. 
            It has become a key driver of innovation in the 21st century.
            """,
        }

    def test_high_originality_short(self):
        text = self.test_texts["High Originality (Short)"]
        analyzer = OriginalityAnalysis(text, reference_texts=self.reference_corpus)
        scores = analyzer.compute_originality_score()
        self.assertGreater(scores["originality_score"], 0.7, "High originality short text should have a high score.")

    def test_moderate_originality_short(self):
        text = self.test_texts["Moderate Originality (Short)"]
        analyzer = OriginalityAnalysis(text, reference_texts=self.reference_corpus)
        scores = analyzer.compute_originality_score()
        self.assertGreater(scores["originality_score"], 0.4, "Moderate originality short text should have a moderate score.")
        self.assertLess(scores["originality_score"], 0.7, "Moderate originality short text should not have a high score.")

    def test_low_originality_short(self):
        text = self.test_texts["Low Originality (Short)"]
        analyzer = OriginalityAnalysis(text, reference_texts=self.reference_corpus)
        scores = analyzer.compute_originality_score()
        self.assertLess(scores["originality_score"], 0.4, "Low originality short text should have a low score.")

    def test_high_originality_medium(self):
        text = self.test_texts["High Originality (Medium)"]
        analyzer = OriginalityAnalysis(text, reference_texts=self.reference_corpus)
        scores = analyzer.compute_originality_score()
        self.assertGreater(scores["originality_score"], 0.7, "High originality medium text should have a high score.")

    def test_moderate_originality_medium(self):
        text = self.test_texts["Moderate Originality (Medium)"]
        analyzer = OriginalityAnalysis(text, reference_texts=self.reference_corpus)
        scores = analyzer.compute_originality_score()
        self.assertGreater(scores["originality_score"], 0.4, "Moderate originality medium text should have a moderate score.")
        self.assertLess(scores["originality_score"], 0.7, "Moderate originality medium text should not have a high score.")

    def test_low_originality_medium(self):
        text = self.test_texts["Low Originality (Medium)"]
        analyzer = OriginalityAnalysis(text, reference_texts=self.reference_corpus)
        scores = analyzer.compute_originality_score()
        self.assertLess(scores["originality_score"], 0.4, "Low originality medium text should have a low score.")

    def test_originality_score_order(self):
        results = {}
        for text_name, test_text in self.test_texts.items():
            analyzer = OriginalityAnalysis(test_text, reference_texts=self.reference_corpus)
            scores = analyzer.compute_originality_score()
            results[text_name] = scores["originality_score"]

        high_scores = [results[key] for key in results if "High Originality" in key]
        moderate_scores = [results[key] for key in results if "Moderate Originality" in key]
        low_scores = [results[key] for key in results if "Low Originality" in key]

        self.assertTrue(max(high_scores) > max(moderate_scores), "High originality texts should have higher scores than moderate originality texts.")
        self.assertTrue(max(moderate_scores) > max(low_scores), "Moderate originality texts should have higher scores than low originality texts.")

if __name__ == '__main__':
    unittest.main()