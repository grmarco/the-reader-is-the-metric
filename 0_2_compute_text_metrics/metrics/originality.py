from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

class TextPreprocessor:
    def __init__(self, text, reference_texts):
        self.text = text
        self.reference_texts = reference_texts

    def validate_inputs(self):
        """
        Checks if the entries are valid.
        """
        if not self.text or not self.reference_texts:
            return False
        return True

from sentence_transformers import util

from sentence_transformers import SentenceTransformer, util
import numpy as np

class EmbeddingAnalysis:
    def __init__(self, embedding_model):
        """
        Initialises the embeddings model based on Sentence Transformers.
        """
        self.embedding_model = embedding_model

    def compute_embeddings(self, texts):
        """
        Calculates embeddings for a list of texts using Sentence Transformers.
        """
        try:
            return self.embedding_model.encode(texts, convert_to_tensor=True)
        except Exception as e:
            print(f"Error in embedding calculation: {e}")
            return None

    def calculate_semantic_distance(self, text_embedding, reference_embeddings):
        """
        Calculates the average semantic distance between the text and the reference texts.
        """
        if text_embedding is None or reference_embeddings is None:
            print("Error in semantic distance calculation: Missing embeddings.")
            return np.nan

        try:
            distances = [
                1 - util.cos_sim(text_embedding, ref_embedding).item()
                for ref_embedding in reference_embeddings
            ]
            return np.mean(distances) if distances else np.nan
        except Exception as e:
            print(f"Error in semantic distance calculation: {e}")
            return np.nan


import torch

class LikelihoodEvaluator:
    def __init__(self, lm_model, lm_tokenizer):
        self.lm_model = lm_model
        self.tokenizer = lm_tokenizer

    def calculate_log_likelihood(self, text):
        """
        Calcula el log-likelihood del texto.
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"]

            with torch.no_grad():
                outputs = self.lm_model(input_ids)
                logits = outputs.logits

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            target_log_probs = log_probs[
                torch.arange(logits.size(0)).unsqueeze(-1),
                torch.arange(logits.size(1)),
                input_ids
            ]
            return target_log_probs.sum().item()
        except Exception as e:
            print(f"Error in log-likelihood calculation: {e}")
            return np.nan

class OriginalityAnalysis:
    def __init__(self, text, reference_texts, lm_model, lm_tokenizer, embedding_model):
        self.preprocessor = TextPreprocessor(text, reference_texts)
        self.embedding_analysis = EmbeddingAnalysis(embedding_model=embedding_model)
        self.likelihood_evaluator = LikelihoodEvaluator(lm_model=lm_model, lm_tokenizer=lm_tokenizer)
        self.text = text
        self.reference_texts = reference_texts

    def compute_originality_score(self):
        if not self.preprocessor.validate_inputs():
            return {
                "semantic_distance": np.nan,
                "log_likelihood": np.nan,
            }

        text_embedding = self.embedding_analysis.compute_embeddings([self.text])
        reference_embeddings = self.embedding_analysis.compute_embeddings(self.reference_texts)

        semantic_distance = self.embedding_analysis.calculate_semantic_distance(
            text_embedding, reference_embeddings
        )
        log_likelihood = self.likelihood_evaluator.calculate_log_likelihood(self.text)


        return {
            "semantic_distance": semantic_distance,
            "log_likelihood": -log_likelihood,
        }
