import math
import numpy as np
from collections import defaultdict
from transformers import pipeline

# ---------------------------------------------------------------------------
# 1) Word processor: Splits into chunks of N words
# ---------------------------------------------------------------------------
class WordChunkProcessor:
    def __init__(self, text, chunk_size=50):
        """
        Splits text into chunks of 'chunk_size' words.
        """
        words = text.split()
        self.chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            self.chunks.append(chunk)

    def get_chunks(self):
        return self.chunks


# ---------------------------------------------------------------------------
# 2) Sentiment Analyser
# ---------------------------------------------------------------------------
class SentimentAnalyzer:
    def __init__(self, sentiment_model_pipeline):
        self.analyzer = sentiment_model_pipeline

    def analyze_sentiment(self, sentences):
        """
        Call the 'sentiment-analysis' pipeline (e.g. roberta).
        Returns a list of dictionaries with sentiment metrics.
        """
        # Model call (truncation=True is relevant if a chunk is very long)
        all_results = self.analyzer(sentences, truncation=True, top_k=None)

        # Mapeo simple: 'positive' => +1, 'negative' => -1
        map_sentiment = {'positive': 1, 'negative': -1}

        sentiment_metrics = []
        for results in all_results:
            # Sometimes huggingface can return a single dict, we force it to list
            if isinstance(results, dict):
                results = [results]

            # Generate dictionary { 'positive': x, 'negative': y }
            scores = {res['label'].lower(): res['score'] for res in results}

            # expected_score = sum de (+1 * positive_score) + (-1 * negative_score)
            expected_score = sum(
                map_sentiment[label] * score
                for label, score in scores.items()
                if label in map_sentiment
            )

            # Entropy of distribution
            entropy = -sum(
                score * math.log(score + 1e-10)
                for score in scores.values()
            )

            max_confidence = max(scores.values())
            main_label = max(scores, key=scores.get)  # 'positive' o 'negative'
            valence_intensity = abs(expected_score)

            sentiment_metrics.append({
                'expected_score': expected_score,
                'entropy': entropy,
                'max_confidence': max_confidence,
                'label': main_label,
                'valence_intensity': valence_intensity
            })

        return sentiment_metrics

    def calculate_text_sentiment_metrics(self, sentences):
        sentiment_metrics = self.analyze_sentiment(sentences)
        if not sentiment_metrics:
            return self._get_empty_metrics()

        metrics = self._calculate_basic_metrics(sentiment_metrics)
        metrics.update(self._calculate_advanced_metrics(sentiment_metrics))
        return metrics

    def _get_empty_metrics(self):
        return {
            'average_sentiment': 0.0,
            'variance_sentiment': 0.0,
            'average_entropy': 0.0,
            'average_max_confidence': 0.0,
            'average_valence_intensity': 0.0,
            'sentiment_distribution_positive': 0.0,
            'sentiment_distribution_negative': 0.0,
            'expected_scores': [],
        }

    def _calculate_basic_metrics(self, sentiment_metrics):
        expected_scores = [m['expected_score'] for m in sentiment_metrics]
        entropies = [m['entropy'] for m in sentiment_metrics]
        max_confidences = [m['max_confidence'] for m in sentiment_metrics]
        valence_intensities = [m['valence_intensity'] for m in sentiment_metrics]
        labels = [m['label'] for m in sentiment_metrics]

        total = len(labels)
        distribution = {
            'positive': labels.count('positive') / total,
            'negative': labels.count('negative') / total
        }

        return {
            'average_sentiment': np.mean(expected_scores),
            'variance_sentiment': np.var(expected_scores),
            'average_entropy': np.mean(entropies),
            'average_max_confidence': np.mean(max_confidences),
            'average_valence_intensity': np.mean(valence_intensities),
            'sentiment_distribution_positive': distribution['positive'],
            'sentiment_distribution_negative': distribution['negative'],
            'expected_scores': expected_scores
        }

    def _calculate_advanced_metrics(self, sentiment_metrics):
        expected_scores = [m['expected_score'] for m in sentiment_metrics]
        trend, coef = self.calculate_sentiment_trend(expected_scores)
        changes = self.calculate_sentiment_changes(expected_scores)

        return {
            'sentiment_trend': trend,
            'sentiment_trend_coefficient': coef.tolist(),
            'sentiment_changes': changes,
            'sentiment_volatility': np.std(changes) if changes else 0
        }

    def calculate_sentiment_trend(self, expected_scores):
        """
        Calculates the slope (trend) of the sentiment scores,
        making a small smoothing with window_size = 3 (or dynamic).
        """
        arr = np.array(expected_scores, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if len(arr) < 2:
            return 0, np.array([0, 0])

        # Adjust window_size dynamically to avoid arrays of length 1 after convolution
        window_size = min(3, len(arr))

        # Smooth or leave as is if the length is small
        if len(arr) == window_size:
            # convolution with mode='valid' would return one value only
            smoothed = arr
        else:
            smoothed = np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

        if len(smoothed) < 2:
            return 0, np.array([0, 0])

        x = np.arange(len(smoothed))
        # If all values are equal, the slope is 0.
        if np.allclose(smoothed, smoothed[0]):
            return 0, np.array([0, 0])

        try:
            coef = np.polyfit(x, smoothed, 1)  # best-fit straight line
        except np.linalg.LinAlgError:
            coef = np.array([0, 0])

        return coef[0], coef

    def calculate_sentiment_changes(self, expected_scores):
        return [
            expected_scores[i] - expected_scores[i - 1]
            for i in range(1, len(expected_scores))
        ]


# ---------------------------------------------------------------------------
# 3) Emotion Analyser
# ---------------------------------------------------------------------------
class EmotionAnalyzer:
    def __init__(self, emotion_model_pipeline, max_possible_emotions=7):
        self.classifier = emotion_model_pipeline
        self.max_possible_emotions = max_possible_emotions

        # Valence mapping for GoEmotions emotions
        self.VALENCE = {
            'love': 1,
            'admiration': 1,
            'approval': 1,
            'joy': 1,
            'neutral': 0,
            'optimism': 1,
            'excitement': 1,
            'desire': 1,
            'realization': 0,
            'gratitude': 1,
            'disapproval': -1,
            'annoyance': -1,
            'disappointment': -1,
            'caring': 1,
            'amusement': 1,
            'sadness': -1,
            'anger': -1,
            'confusion': -1,      
            'surprise': 0,         
            'curiosity': 0,        
            'disgust': -1,
            'remorse': -1,
            'fear': -1,
            'pride': 1,
            'relief': 1,
            'nervousness': -1,
            'embarrassment': -1,
            'grief': -1
        }

    def analyze_emotions(self, sentences):
        """
        Batch analyse the emotions of each chunk of text.
        """
        all_results = self.classifier(sentences, truncation=True, top_k=None)

        emotion_scores = []
        for results in all_results:
            if isinstance(results, list) and len(results) > 0:
                emotion_dict = {item['label']: item['score'] for item in results}
                emotion_scores.append(emotion_dict)
            else:
                raise ValueError("Unexpected emotion pipeline results structure")

        return emotion_scores

    def calculate_emotional_metrics(self, sentences):
        """
        Calculate aggregate metrics (emotional_richness, temporal_dynamics, complexity)
        as well as transitions and new indices (volatility, persistence, oscillation).
        """
        emotion_trajectory = self.calculate_emotion_dynamics(sentences)
        richness = self.calculate_emotional_richness(emotion_trajectory)
        dynamics = self.calculate_temporal_dynamics(emotion_trajectory)
        complexity = self.calculate_emotional_complexity(emotion_trajectory)

        # Aquí calculamos transiciones y sus índices
        transitions_map = self._calculate_emotion_transitions_map(emotion_trajectory)
        transition_indices = self._calculate_transition_indices(transitions_map)

        return {
            'emotion_trajectory': dict(emotion_trajectory),
            'emotional_richness': richness,
            'temporal_dynamics': dynamics,
            'emotional_complexity': complexity,

            # Transitions map (dictionary {emotion_from: {emotion_to: count}})
            'emotion_transitions_map': transitions_map,

            # New indices
            'emotional_volatility': transition_indices['emotional_volatility'],
            'emotional_persistence': transition_indices['emotional_persistence'],
            'emotional_oscillation': transition_indices['emotional_oscillation']
        }

    def calculate_emotion_dynamics(self, sentences):
        """
        Returns a dictionary: {emotion: [intensity in chunk 0, chunk 1, ...]}.
        """
        emotion_scores = self.analyze_emotions(sentences)
        trajectory = defaultdict(list)
        for emotion_dict in emotion_scores:
            for emotion, value in emotion_dict.items():
                trajectory[emotion].append(value)
        return trajectory

    def calculate_emotional_richness(self, emotion_trajectory, threshold=0.3):
        """
        Calculate how many emotions appear above 'threshold' at some point.
        Returns a value between 0 and 1, normalised by 'max_possible_emotions'.
        """
        if not emotion_trajectory:
            return 0.0

        emotions_present = sum(
            1 for values in emotion_trajectory.values()
            if any(v > threshold for v in values)
        )
        return emotions_present / self.max_possible_emotions

    def calculate_temporal_dynamics(self, emotion_trajectory, threshold=0.5):
        """
        It measures changes in the intensity of emotions (variance) and
        how many times the dominant emotion changes (transitions).
        It combines both into an approximate 0..1 index.
        """
        if not emotion_trajectory:
            return 0.0

        dominant_emotions = self._get_dominant_emotions(emotion_trajectory, threshold)
        transitions = self._count_transitions(dominant_emotions)
        intensity_variance = self._calculate_intensity_variance(emotion_trajectory)

        return 0.5 * intensity_variance + 0.5 * transitions

    def calculate_emotional_complexity(self, emotion_trajectory):
        """
        Aggregate entropy (in base 2) of the intensity matrix.
        The higher, the more 'complex' the mix of emotions.
        """
        if not emotion_trajectory:
            return 0.0

        # We convert to numpy 2D array: each row an emotion, each column a chunk.
        all_values = list(emotion_trajectory.values())  
        mat = np.array(all_values, dtype=float)

        # Calculate entropy: -sum(p * log2(p))
        total_sum = np.sum(mat)
        if total_sum == 0:
            return 0.0

        normalized = mat / total_sum
        # Avoid log(0)
        entropy = -np.sum(normalized * np.log2(normalized + 1e-12))
        # Divide by log2(#elements) to normalise
        max_possible = np.log2(mat.size)
        return float(entropy / max_possible)

    def _get_dominant_emotions(self, trajectory, threshold):
        """
        Returns the dominant emotion in each chunk if it exceeds the threshold, otherwise None.
        """
        if not trajectory:
            return []
        from itertools import zip_longest

        list_lengths = [len(lst) for lst in trajectory.values()]
        min_len = min(list_lengths) if list_lengths else 0

        dominant = []
        for i in range(min_len):
            best_emo = None
            best_score = float("-inf")
            for emotion, values in trajectory.items():
                if values[i] > best_score:
                    best_score = values[i]
                    best_emo = emotion
            if best_score > threshold:
                dominant.append(best_emo)
            else:
                dominant.append(None)
        return dominant

    def _count_transitions(self, dominant_emotions):
        """
        Proporción de cambios de emoción dominante vs. transiciones validas.
        """
        transitions = 0
        valid_pairs = 0
        for i in range(1, len(dominant_emotions)):
            prev_emo = dominant_emotions[i - 1]
            curr_emo = dominant_emotions[i]
            if prev_emo is not None and curr_emo is not None:
                valid_pairs += 1
                if prev_emo != curr_emo:
                    transitions += 1
        return transitions / valid_pairs if valid_pairs > 0 else 0

    def _calculate_intensity_variance(self, trajectory):
        """
        Varianza promedio de las intensidades de cada emoción a través de los chunks.
        """
        variances = [np.var(values) for values in trajectory.values()]
        return float(np.mean(variances)) if variances else 0.0

    # -----------------------------------------------------------------------
    #  AQUÍ VIENE LA PARTE NUEVA: MAPA DE TRANSICIONES + ÍNDICES
    # -----------------------------------------------------------------------
    def _calculate_emotion_transitions_map(self, emotion_trajectory):
        """
        Retorna un dict con las transiciones reales entre chunks consecutivos.
        - Calcula la emoción dominante en cada chunk.
        - Suma 1 en transitions_map[from_emo][to_emo] por cada cambio consecutivo.
        """
        from collections import defaultdict

        # Comprobamos si hay datos
        if not emotion_trajectory:
            return {}

        # 1) Hallar cuántos chunks (mínimo de longitudes)
        lengths = [len(v) for v in emotion_trajectory.values()]
        num_chunks = min(lengths) if lengths else 0
        if num_chunks < 2:
            return {}

        # 2) Emoción dominante en cada chunk (sin umbral, para ver siempre su cambio).
        dominant_emotion_per_chunk = []
        for i in range(num_chunks):
            best_emo = None
            best_score = float("-inf")
            for emotion, values in emotion_trajectory.items():
                if values[i] > best_score:
                    best_score = values[i]
                    best_emo = emotion
            dominant_emotion_per_chunk.append(best_emo)

        # 3) Construir el mapa de transiciones
        transitions_map = defaultdict(lambda: defaultdict(int))
        for i in range(num_chunks - 1):
            from_emo = dominant_emotion_per_chunk[i]
            to_emo = dominant_emotion_per_chunk[i + 1]
            transitions_map[from_emo][to_emo] += 1

        # Convertir a dict normal
        transitions_map = {k: dict(v) for k, v in transitions_map.items()}
        return transitions_map

    def _calculate_transition_indices(self, transitions_map):
        """
        A partir del dict transitions_map = {'joy': {'sadness': x, ...}, ...},
        calculamos:
         - emotional_volatility: proporción de cambios E->X con E != X
         - emotional_persistence: proporción E->E
         - emotional_oscillation: proporción pos->neg o neg->pos
        """
        if not transitions_map:
            return {
                'emotional_volatility': 0.0,
                'emotional_persistence': 0.0,
                'emotional_oscillation': 0.0
            }

        total_transitions = 0
        changes_in_dominance = 0  # E->X con E != X
        cross_valence = 0         # pos->neg o neg->pos

        for from_emo, to_dict in transitions_map.items():
            from_val = self.VALENCE.get(from_emo, 0)
            for to_emo, count in to_dict.items():
                to_val = self.VALENCE.get(to_emo, 0)

                total_transitions += count
                if from_emo != to_emo:
                    changes_in_dominance += count

                # Valence crossing: +1 => -1 o -1 => +1
                if (from_val * to_val) == -1:
                    cross_valence += count

        if total_transitions == 0:
            return {
                'emotional_volatility': 0.0,
                'emotional_persistence': 0.0,
                'emotional_oscillation': 0.0
            }

        emotional_volatility = changes_in_dominance / total_transitions
        emotional_persistence = (total_transitions - changes_in_dominance) / total_transitions
        emotional_oscillation = cross_valence / total_transitions

        return {
            'emotional_volatility': emotional_volatility,
            'emotional_persistence': emotional_persistence,
            'emotional_oscillation': emotional_oscillation
        }


# ---------------------------------------------------------------------------
# 4) Clase principal: combina análisis de sentimiento y emociones
# ---------------------------------------------------------------------------
class SentimentAnalysis:
    def __init__(self, text, sentiment_pipeline, emotion_pipeline, chunk_size=50):
        # En lugar de oraciones, dividimos en chunks de 'chunk_size' palabras
        self.text_processor = WordChunkProcessor(text, chunk_size=chunk_size)
        self.sentiment_analyzer = SentimentAnalyzer(sentiment_pipeline)
        self.emotion_analyzer = EmotionAnalyzer(emotion_pipeline)

    # Método que unifica los resultados de sentimiento y emociones
    def compute_sentiment_analysis(self):
        chunks = self.text_processor.get_chunks()

        # Métricas de sentimiento
        sentiment_metrics = self.sentiment_analyzer.calculate_text_sentiment_metrics(chunks)

        # Métricas de emoción, que ahora incluyen los nuevos índices
        emotion_metrics = self.emotion_analyzer.calculate_emotional_metrics(chunks)

        # Construyes el diccionario final, tomando los campos que te interesen
        result = {
            # Métricas de sentimiento
            'average_sentiment': sentiment_metrics['average_sentiment'],
            'variance_sentiment': sentiment_metrics['variance_sentiment'],
            'average_entropy': sentiment_metrics['average_entropy'],
            'average_max_confidence': sentiment_metrics['average_max_confidence'],
            'average_valence_intensity': sentiment_metrics['average_valence_intensity'],
            'sentiment_distribution_positive': sentiment_metrics['sentiment_distribution_positive'],
            'sentiment_distribution_negative': sentiment_metrics['sentiment_distribution_negative'],
            'expected_scores': sentiment_metrics['expected_scores'],
            'sentiment_trend': sentiment_metrics['sentiment_trend'],
            'sentiment_trend_coefficient': sentiment_metrics['sentiment_trend_coefficient'],
            'sentiment_changes': sentiment_metrics['sentiment_changes'],
            'sentiment_volatility': sentiment_metrics['sentiment_volatility'],

            # Métricas de emoción
            'emotion_trajectory': emotion_metrics['emotion_trajectory'],
            'emotional_richness': emotion_metrics['emotional_richness'],
            'temporal_dynamics': emotion_metrics['temporal_dynamics'],
            'emotional_complexity': emotion_metrics['emotional_complexity'],

            # Incorporar el mapa de transiciones, si lo deseas
            'emotion_transitions_map': emotion_metrics['emotion_transitions_map'],

            # Y los NUEVOS ÍNDICES:
            'emotional_volatility': emotion_metrics['emotional_volatility'],
            'emotional_persistence': emotion_metrics['emotional_persistence'],
            'emotional_oscillation': emotion_metrics['emotional_oscillation']
        }

        return result