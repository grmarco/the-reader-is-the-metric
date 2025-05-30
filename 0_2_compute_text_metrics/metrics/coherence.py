from sentence_transformers import util
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import numpy as np
from scipy.spatial.distance import cosine

def validate_distribution(vector):
    if np.any(vector < 0):
        return False
    if np.isnan(vector).any() or np.isinf(vector).any():
        return False
    if not np.isclose(vector.sum(), 1):
        return False
    return True


class TextSplitter:
    @staticmethod
    def split_into_sections(text, words_per_section):
        """
        Splits the text into sections of a fixed number of words.
        """
        words = [word.strip() for word in text.split() if word.strip()]
        sections = [
            " ".join(words[i:i + words_per_section])
            for i in range(0, len(words), words_per_section)
        ]
        return sections if len(sections) > 1 else []  # Devuelve [] si no hay suficientes secciones

class LocalCoherenceAnalyzer:
    def __init__(self, sentences, embedding_model):
        self.sentences = sentences
        # self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.embedding_model = embedding_model
    
    def compute_embeddings(self, text_list):
        """Encodes a list of texts into embeddings."""
        return self.embedding_model.encode(text_list, convert_to_tensor=True)
    
    def measure_local_coherence(self):
        """
        Calculates local coherence based on cosine similarity between consecutive sentences.
        Penalizes if the similarity is consistently high (indicative of repetition).
        """
        if not self.sentences or len(self.sentences) < 2:
            return 0  # Devuelve 0 si no hay suficientes frases

        try:
            embeddings = self.compute_embeddings(self.sentences)
            similarities = [
                util.cos_sim(embeddings[i], embeddings[i + 1]).item()
                for i in range(len(embeddings) - 1)
            ]
            return np.mean(similarities) if similarities else np.nan
        except Exception as e:
            print(f"Error in local coherence calculation: {e}")
            return np.nan

class EntityGridCoherenceAnalyzer:
    def __init__(self, text, spacy_model):
        """
        text: str, the text to analyze.
        spacy_model: a loaded spaCy nlp object (e.g., spacy.load('en_core_web_sm'))
        """
        self.text = text
        self.nlp = spacy_model
        
        # Analizamos el texto con spaCy
        self.doc = self.nlp(self.text)
        # Guardamos las oraciones
        self.sentences = list(self.doc.sents)
        
        # Construimos el entity grid en un DataFrame
        self.entity_df = self._build_entity_grid()

    def _build_entity_grid(self):
        """
        Returns a DataFrame where:
          - rows = sentences
          - columns = entities (lowercase strings)
          - values = role ('S', 'O', 'X') or '-' if the entity does not appear in that sentence
        """
        all_rows = []
        global_entities = set()

        # First, we build a "dictionary per sentence" representation: {entity: role}
        for sent in self.sentences:
            sent_dict = {}
            for token in sent:
                if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                    # Determinamos rol
                    if token.dep_ in ["nsubj", "nsubj:pass"]:
                        role = "S"
                    elif token.dep_ in ["obj", "iobj", "dobj"]:
                        role = "O"
                    else:
                        role = "X"

                    entity_name = token.lemma_.lower()
                    sent_dict[entity_name] = role
                    global_entities.add(entity_name)

            all_rows.append(sent_dict)

        # Convertimos la lista de dicts a un DF
        sorted_entities = sorted(global_entities)
        matrix = []
        for row_dict in all_rows:
            row_vals = []
            for ent in sorted_entities:
                row_vals.append(row_dict.get(ent, "-"))
            matrix.append(row_vals)
        
        entity_df = pd.DataFrame(matrix, columns=sorted_entities)
        return entity_df
    
    def measure_entity_coherence(self):
        """
        Calculates a simple local coherence metric based on the Entity Grid.
        Returns a higher value if there is more continuity of entities between consecutive sentences.
        """
        df = self.entity_df
        num_sentences = df.shape[0]
        if num_sentences < 2:
            return 0.0  # Not enough sentences
        
        total_pairs = num_sentences - 1
        pair_scores = []

        for i in range(total_pairs):
            row1 = df.iloc[i]
            row2 = df.iloc[i+1]

            # Entities that appear in at least one of the two sentences
            ents_row1 = set(row1.index[row1 != "-"])
            ents_row2 = set(row2.index[row2 != "-"])
            entities_in_both = ents_row1.union(ents_row2)

            if len(entities_in_both) == 0:
                pair_scores.append(0)
                continue

            pair_score = 0
            for ent in entities_in_both:
                role1 = row1[ent]
                role2 = row2[ent]
                # If the entity appears in both sentences...
                if role1 != "-" and role2 != "-":
                    # Give a point for the mere continuity of the same entity
                    pair_score += 1
                    # Bonus if the role remains the same (S->S, O->O, X->X)
                    if role1 == role2:
                        pair_score += 0.5

            # Normalize by the number of entities involved
            pair_score /= len(entities_in_both)
            pair_scores.append(pair_score)

        return np.mean(pair_scores) if pair_scores else 0.0


from scipy.spatial.distance import jensenshannon

class GlobalCoherenceAnalyzer:
    def __init__(self, sections):
        """
        sections: lista de strings, cada string es una 'sección' del texto
        """
        self.sections = sections

    def find_optimal_num_topics(self, tokenized_texts, start=2, limit=10, step=1, passes=20, coherence_type='c_v'):
        """
        Explores different values of num_topics and selects the best one 
        according to the topic coherence metric (default is 'c_v').
        
        Returns:
        - best_num_topics
        - best_coherence
        - dict with {num_topics -> coherence_score}
        """
        dictionary = Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        best_num_topics = None
        best_coherence = -np.inf
        coherence_values = {}

        for num_topics in range(start, limit + 1, step):
            lda_model = LdaModel(
                corpus=corpus,
                num_topics=num_topics,
                id2word=dictionary,
                passes=passes,
                random_state=42
            )
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=tokenized_texts,
                dictionary=dictionary,
                coherence=coherence_type
            )
            score = coherence_model.get_coherence()
            coherence_values[num_topics] = score

            if score > best_coherence:
                best_coherence = score
                best_num_topics = num_topics

        return best_num_topics, best_coherence, coherence_values

    def topic_modeling_global_coherence(
        self,
        num_topics=None,
        start=2,
        limit=10,
        step=1,
        passes=20,
        coherence_type='c_v'
    ):
        """
        Unified global coherence metric:
         1) If num_topics is None, automatically find the best one in [start..limit].
         2) Train LDA with the chosen num_topics.
         3) Calculate two types of similarity (dot product):
            - overall_coherence: between all pairs of sections
            - consecutive_coherence: only between consecutive sections
         4) Calculate the Jensen-Shannon divergence between topic distributions.
         5) Return these values, along with (best_num_topics, coherence_values).
        """
        if not self.sections or len(self.sections) < 2:
            return {
                "overall_coherence": None,
                "consecutive_coherence": None,
                "best_num_topics": None,
                "coherence_values": {}
            }

        try:
            # tokenizing
            tokenized_sections = [section.split() for section in self.sections]
            dictionary = Dictionary(tokenized_sections)
            corpus = [dictionary.doc2bow(text) for text in tokenized_sections]

            # 1) Determining number of topics (automatically or manually)
            if num_topics is not None:
                best_num_topics = num_topics
                coherence_values = {}
            else:
                best_num_topics, _, coherence_values = self.find_optimal_num_topics(
                    tokenized_texts=tokenized_sections,
                    start=start,
                    limit=limit,
                    step=step,
                    passes=passes,
                    coherence_type=coherence_type
                )

    
            # 2) Traning the LDA model
            lda_model = LdaModel(
                corpus=corpus,
                num_topics=best_num_topics,
                id2word=dictionary,
                passes=passes,
                random_state=42
            )

            # 3) Get topic distributions
            topic_distributions = [
                lda_model.get_document_topics(bow, minimum_probability=0)
                for bow in corpus
            ]
            topic_vectors = np.array([
                [prob for _, prob in dist]
                for dist in topic_distributions
            ])

            # normalizing topic vectors
            topic_vectors_sum = topic_vectors.sum(axis=1, keepdims=True)
            # Evitar división por cero
            topic_vectors_sum[topic_vectors_sum == 0] = 1
            topic_vectors = topic_vectors / topic_vectors_sum

            # Validación de las distribuciones
            valid_indices = []
            for idx, vec in enumerate(topic_vectors):
                if validate_distribution(vec):
                    valid_indices.append(idx)
                else:
                    print(f"Distribución inválida en la sección {idx}: {vec}")

            # Filtrar solo las distribuciones válidas
            topic_vectors = topic_vectors[valid_indices]

            # Verificar si hay suficientes distribuciones válidas
            if len(topic_vectors) < 2:
                print("No hay suficientes distribuciones válidas para calcular la coherencia.")
                return {
                    "overall_coherence": np.nan,
                    "best_num_topics": best_num_topics,
                    "coherence_values": coherence_values
                }

            # overall_coherence: similitud coseno entre todos los pares de secciones
            similarities = []
            num_sections = len(topic_vectors)
            for i in range(num_sections):
                for j in range(i + 1, num_sections):
                    sim = 1 - cosine(topic_vectors[i], topic_vectors[j])
                    if not np.isnan(sim):
                        similarities.append(sim)
            overall_coherence = np.mean(similarities) if similarities else 0.0    

            # 4) Retornamos los resultados sin penalizaciones
            return {
                "overall_coherence": overall_coherence,
                "best_num_topics": best_num_topics,
                "coherence_values": coherence_values
            }

        except Exception as e:
            print(f"[ERROR] GlobalCoherenceAnalyzer: {e}")
            return {
                "overall_coherence": np.nan,
                "consecutive_coherence": np.nan,
                "best_num_topics": None,
                "coherence_values": {}
            }

class CoherenceAnalysis:
    def __init__(self, text, embedding_model, spacy_model, words_per_section=5):
        """
        New version of CoherenceAnalysis using the GlobalCoherenceAnalyzer class.
        """
        self.text = text
        self.sentences_per_section = words_per_section

        # Text sections for global analysis
        self.sections = TextSplitter.split_into_sections(text, words_per_section)
        
        # Local coherence analyzer with embeddings
        self.local_analyzer = LocalCoherenceAnalyzer(text.split("."), embedding_model)
        
        # Local coherence analyzer with Entity Grid
        self.entity_grid_analyzer = EntityGridCoherenceAnalyzer(text, spacy_model)

        # Unified global coherence analyzer
        self.global_analyzer = GlobalCoherenceAnalyzer(self.sections)

    def compute_coherence(
        self,
        num_topics=5,
        start=2,
        limit=10,
        step=1,
        passes=20,
        coherence_type='c_v'
    ):
        """
        Combines multiple coherence metrics:
          - Local coherence by embeddings (local_coherence_embeddings)
          - Local coherence by entity grid (entity_coherence)
          - Unified global coherence (overall_coherence, consecutive_coherence, best_num_topics, coherence_values)
        """
        # Local coherence metric with embeddings
        local_coherence_embeddings = self.local_analyzer.measure_local_coherence()
        
        # Local coherence metric with entity grid
        entity_coherence = self.entity_grid_analyzer.measure_entity_coherence()
        
        # Unified global coherence (GlobalCoherenceAnalyzer)
        global_results = self.global_analyzer.topic_modeling_global_coherence(
            num_topics=num_topics,
            start=start,
            limit=limit,
            step=step,
            passes=passes,
            coherence_type=coherence_type
        )

        return {
            "local_coherence_embeddings": local_coherence_embeddings,
            "entity_coherence": entity_coherence,
            "overall_coherence": global_results.get("overall_coherence"),
            "best_num_topics": global_results.get("best_num_topics"),
            "coherence_values": global_results.get("coherence_values"),
        }