superprompt = """You are an expert in literary analysis. Analyze the following text and identify literary devices. Include the following categories, with their definitions and examples provided below:

1. **Metaphors**: A figure of speech that describes an object or action as if it were something else, without using "like" or "as".
   - Example: "Time is a thief that steals our moments."

2. **Similes**: A figure of speech comparing two different things using "like" or "as".
   - Example: "Her smile was as bright as the sun."

3. **Alliteration**: The repetition of the same initial consonant sound in closely placed words.
   - Example: "The whispering wind wandered through the woods."

4. **Irony**: A figure of speech where the intended meaning is different from the literal meaning, often highlighting contradictions.
   - Example: "The fire station burned down."

5. **Personification**: Attributing human qualities to non-human objects or abstract ideas.
   - Example: "The trees danced in the wind."

6. **Hyperbole**: Exaggeration for emphasis or effect.
   - Example: "I’ve told you a million times!"

7. **Oxymoron**: A combination of two contradictory terms.
   - Example: "A deafening silence filled the room."

8. **Synesthesia**: A literary device that combines senses to describe something in an unusual way.
   - Example: "The music tasted sweet."

9. **Anaphora**: Repetition of a word or phrase at the beginning of successive clauses or sentences.
   - Example: "We shall fight on the beaches, we shall fight on the landing grounds."

10. **Onomatopoeia**: Words that imitate sounds.
    - Example: "The bees buzzed in the garden."

11. **Paradox**: A seemingly contradictory statement that reveals a deeper truth.
    - Example: "Less is more."

12. **Rhetorical Questions**: Questions asked for effect or to emphasize a point rather than to get an answer.
    - Example: "Is this not the best day of your life?"

13. **Euphony**: Use of pleasant, harmonious sounds.
    - Example: "The mellow waves lapped gently at the shore."

14. **Enumerations**: Listing details or elements to emphasize an idea.
    - Example: "She brought pencils, pens, paper, and notebooks."

15. **Allusions**: Indirect references to well-known events, people, places, or works of literature.
    - Example: "He was her Romeo, and she was his Juliet."

16. **Sarcasm**: A form of verbal irony intended to mock or convey contempt.
    - Example: "Oh great, another homework assignment!"

17. **Allegory**: A narrative in which characters, events, and details represent abstract ideas or moral concepts.
    - Example: "George Orwell’s Animal Farm is an allegory for the Russian Revolution."

18. **Symbolism**: The use of symbols to represent ideas or qualities.
    - Example: "The dove is a symbol of peace."

19. **Epiphora**: Repetition of a word or phrase at the end of successive clauses or sentences.
    - Example: "I want pizza, he wants pizza, we all want pizza."

20. **Paronomasia**: A play on words using similar-sounding words with different meanings (pun).
    - Example: "A bicycle can’t stand on its own because it’s two-tired."

### Your task:
Analyze the following text and identify any instances of these literary devices. Provide the output in the following structured JSON format:

{
    "metaphors": ["<detected metaphor>", ...],
    "similes": ["<detected simile>", ...],
    "alliteration": ["<detected alliteration>", ...],
    "irony": ["<detected irony>", ...],
    "personification": ["<detected personification>", ...],
    "hyperbole": ["<detected hyperbole>", ...],
    "oxymoron": ["<detected oxymoron>", ...],
    "synesthesia": ["<detected synesthesia>", ...],
    "anaphora": ["<detected anaphora>", ...],
    "onomatopoeia": ["<detected onomatopoeia>", ...],
    "paradox": ["<detected paradox>", ...],
    "rhetorical_questions": ["<detected rhetorical question>", ...],
    "euphony": ["<detected euphony>", ...],
    "enumerations": ["<detected enumeration>", ...],
    "allusions": ["<detected allusion>", ...],
    "sarcasm": ["<detected sarcasm>", ...],
    "symbolism": ["<detected symbolism>", ...],
    "epiphora": ["<detected epiphora>", ...],
    "paronomasia": ["<detected paronomasia>", ...]
}

Text:
<<text>>

Output the structured JSON:"""

import openai
import json

class LiteraryDeviceDetector:
    def __init__(self, api_key):
        """
        Initialises the class with the OpenAI API key.
        """
        self.api_key = api_key
        self.client = openai.Client(api_key=api_key)


    def generate_response(self, prompt, model="gpt-4o", temperature=0.2):
        """
        Generate a response using the OpenAI API.
        """
        messages = [
            {"role": "system", "content": "You are an expert in literary analysis."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,                
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] OpenAI API request failed: {e}")
            return None

    def detect_literary_devices(self, text):
        """
        Detects literary resources in the text and organises them in a structured format.
        """
        prompt = superprompt.replace('<<text>>',text)
        response = self.generate_response(prompt).replace('```json','').replace('```','')

        try:
            literary_devices = json.loads(response)
            return literary_devices
        except json.JSONDecodeError:
            print("[ERROR] Failed to parse the model's response.")
            print("[MODEL OUTPUT]:", response)
            return None

    def quantify_devices(self, literary_devices):
        """
        Quantify the quantity and variety of literary resources detected.
        """
        counts = {key: len(value) for key, value in literary_devices.items()}
        total_count = sum(counts.values())
        variety = len([key for key, value in counts.items() if value > 0])
        return {"counts": counts, "total_count": total_count, "variety": variety}
    

import spacy


class SyntacticComplexityAnalyzer:
    def __init__(self, text, spacy_model="en_core_web_sm"):
        """
        Initialises the parsing of syntactic complexity with a spaCy model.
        """
        self.text = text
        # Cargamos el modelo de spaCy (o recibimos la instancia ya cargada)
        # Si ya tienes un nlp inicializado, simplemente haz: self.nlp = spacy_model
        if isinstance(spacy_model, str):
            self.nlp = spacy.load(spacy_model)
        else:
            self.nlp = spacy_model

        self.doc = self.nlp(self.text)

    def average_sentence_length(self):
        """
        Calculates the average length of sentences in the text (in tokens).
        """
        sentences = list(self.doc.sents)
        lengths = [len(sent) for sent in sentences]
        return np.mean(lengths) if lengths else 0

    def count_subordinate_clauses(self):
        """
        Count the number of subordinate clauses in the text.
        (e.g. tokens with dep_ == 'mark' or others).
        """
        # "mark" y otros labels como "advcl", "relcl", etc.
        # Solo como conteo total, sin anidación
        subordinate_labels = {"mark", "advcl", "relcl", "ccomp", "xcomp"}
        subordinate_clauses = sum(1 for token in self.doc if token.dep_ in subordinate_labels)
        return subordinate_clauses

    def dependency_variety(self):
        """
        Measures the variety of dependency relationships in the text.
        """
        dependencies = [token.dep_ for token in self.doc]
        unique_dependencies = set(dependencies)
        return len(unique_dependencies)

    def sentence_complexity_index(self):
        """
        Calculates a complexity index based on the number of subordinate clauses per sentence.
        """
        sentences = list(self.doc.sents)
        subordinate_labels = {"mark", "advcl", "relcl", "ccomp", "xcomp"}
        num_clauses = sum(1 for token in self.doc if token.dep_ in subordinate_labels)
        return num_clauses / len(sentences) if sentences else 0

    def _compute_tree_depth(self, token):
        """
        Calculates the depth of a dependency subtree from a token.
        If the token has no children, returns 1.
        """
        children = list(token.children)
        if not children:
            return 1
        try:
            return 1 + max(self._compute_tree_depth(child) for child in children)
        except ValueError:  # Manejo de casos con listas vacías
            return 1

    def average_tree_depth(self):
        """
        Calculates the average depth of dependency trees in the text (does not distinguish whether it is subordination or other type).
        """
        try:
            depths = [self._compute_tree_depth(token) for token in self.doc]
            return np.mean(depths) if depths else 0
        except Exception as e:
            print(f"[ERROR] in average_tree_depth: {e}")
            return 0

    ##########################################################################
    # Calculation of nesting of subordinations
    ##########################################################################

    def _subordination_depth(self, token, subordination_labels):
        """
        Returns 0 if the token does not induce subordination.
        If it does induce subordination, we recursively scan the children to see if there are
        if there are more nested subordinations, and add 1 for each level.
        """
        if token.dep_ not in subordination_labels:
            return 0

        max_child_depth = 0
        for child in token.children:
            child_depth = self._subordination_depth(child, subordination_labels)
            if child_depth > max_child_depth:
                max_child_depth = child_depth

        # 1 representa el nivel de subordinación del token actual
        return 1 + max_child_depth

    def measure_nested_subordination(self):
        """
        Traverses through all tokens and calculates the nesting 'level' of subordinations
        for each token that is a subordinator.

        Returns dict with:
          - max_subordination_depth: maximum level in the whole document.
          - average_subordination_depth: average of levels for all subordinating tokens
        """
        subordination_labels = {"mark", "advcl", "relcl", "ccomp", "xcomp"}

        depths = []
        for token in self.doc:
            d = self._subordination_depth(token, subordination_labels)
            if d > 0:
                depths.append(d)

        if not depths:
            return {
                "max_subordination_depth": 0,
                "average_subordination_depth": 0
            }

        return {
            "max_subordination_depth": max(depths),
            "average_subordination_depth": np.mean(depths)
        }

    def analyze(self):
        """
        Runs all parsing metrics and returns a summary.
        """
        avg_length = self.average_sentence_length()
        num_subordinate_clauses = self.count_subordinate_clauses()
        dependency_var = self.dependency_variety()
        complexity_index = self.sentence_complexity_index()
        avg_tree_d = self.average_tree_depth()
        nested_subordination = self.measure_nested_subordination()

        return {
            "average_sentence_length": avg_length,
            "num_subordinate_clauses": num_subordinate_clauses,
            "dependency_variety": dependency_var,
            "sentence_complexity_index": complexity_index,
            "average_tree_depth": avg_tree_d,
            "max_subordination_depth": nested_subordination["max_subordination_depth"],
            "average_subordination_depth": nested_subordination["average_subordination_depth"]
        }



from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import entropy

class SemanticDensityAnalyzer:
    def __init__(self, text, embedding_model):
        self.text = text
        self.embedding_model = embedding_model
        self.words = self.tokenize()

    def tokenize(self):
        return [word.strip() for word in self.text.split() if word.strip()]

    def compute_word_embeddings(self):
        return self.embedding_model.encode(self.words)

    def calculate_cosine_similarity(self):
        embeddings = self.compute_word_embeddings()
        similarities = [
            cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0, 0]
            for i in range(len(embeddings) - 1)
        ]
        return np.mean(similarities) if similarities else 0

    def calculate_entropy(self):
        word_frequencies = np.array([self.words.count(word) for word in set(self.words)])
        probabilities = word_frequencies / word_frequencies.sum()
        return entropy(probabilities)

    def analyze(self):
        return {
            "cosine_similarity": self.calculate_cosine_similarity(),
            "entropy": self.calculate_entropy(),
        }


import re
import math
import nltk
import numpy as np
from statistics import stdev, mean
from itertools import accumulate


##############################################################################
# 1. LexicalDiversityAnalyzer
##############################################################################

class LexicalDiversityAnalyzer:
    """
    Calculates various lexical diversity metrics: TTR, log-TTR, Herdan's C and MTLD.
    """

    def __init__(self, tokens):
        """
        :param tokens: list of tokens (strings) already cleaned (without stopwords if you want)
        """
        self.tokens = tokens
        self.num_tokens = len(tokens)
        self.unique_tokens = len(set(tokens))
    
    def ttr(self):
        """
        Type-Token Ratio = unique_tokens / total_tokens
        """
        if self.num_tokens == 0:
            return 0
        return self.unique_tokens / self.num_tokens
    
    def log_ttr(self):
        """
        log-TTR = log(unique_tokens) / log(total_tokens)
        """
        if self.num_tokens <= 1 or self.unique_tokens <= 1:
            return 0
        return math.log(self.unique_tokens) / math.log(self.num_tokens)
    
    def mtld(self, ttr_threshold=0.72):
        """
        Measure of Textual Lexical Diversity (MTLD).

        Standard approach:
        - The sequence of tokens is traversed, TTR is accumulated and each time TTR < ttr_threshold.
          the count is restarted.
        - This is done for the normal sequence and the inverted sequence, and averaged.

        Note: There are several implementations and variants. This is a common approach.
        """

        if self.num_tokens == 0:
            return 0
        
        def mtld_calc(seq):
            factors_count = 0
            token_count = 0
            types = set()
            
            for token in seq:
                token_count += 1
                types.add(token)
                current_ttr = len(types) / token_count
                if current_ttr <= ttr_threshold:
                    factors_count += 1
                    # reset
                    token_count = 0
                    types = set()
            # Ajuste final
            remainder = 1.0 - (current_ttr / ttr_threshold)
            factors_count += remainder
            return len(seq) / factors_count
        
        forward_mtld = mtld_calc(self.tokens)
        reverse_mtld = mtld_calc(self.tokens[::-1])
        return (forward_mtld + reverse_mtld) / 2
    
    def compute_all(self):
        """
        Returns a dict with all calculated metrics.
        """
        return {
            "TTR": self.ttr(),
            "log_TTR": self.log_ttr(),
            "MTLD": self.mtld()
        }


##############################################################################
# 2. SentenceRhythmAnalyzer
##############################################################################

class SentenceRhythmAnalyzer:
    """
    Measures the variation (rhythm) of sentence length throughout the text.
    It can be measured in 'tokens' or in 'syllables'.
    For simplicity, we use 'tokens' here.
    """

    def __init__(self, text, spacy_nlp):
        """
        :param text: full text
        :param spacy_nlp: nlp instance of spaCy with loaded model
        """
        self.text = text
        self.spacy_nlp = spacy_nlp
        # Parseamos el texto para obtener frases y tokens
        self.sentences = list(self.spacy_nlp(self.text).sents)

    def get_sentence_lengths(self):
        """
        Returns the list of lengths of each phrase in tokens.
        """
        lengths = [len(sent) for sent in self.sentences]
        return lengths

    def analyze_rhythm(self):
        """
        Calculate:
        - average_sentence_length
        - std_dev_sentence_length
        - average_change (average difference between consecutive lengths)
        - coefficient_of_variation (std / mean)

        Returns a dict with the results.
        """
        lengths = self.get_sentence_lengths()
        if not lengths:
            return {
                "average_sentence_length": 0,
                "std_dev_sentence_length": 0,
                "average_change": 0,
                "coefficient_of_variation": 0
            }
        
        avg_len = mean(lengths)
        std_len = stdev(lengths) if len(lengths) > 1 else 0
        
        # Calcular la diferencia promedio entre oraciones consecutivas
        diffs = [abs(lengths[i+1] - lengths[i]) for i in range(len(lengths)-1)]
        avg_change = mean(diffs) if diffs else 0
        
        # Coef. de variación
        coeff_var = std_len / avg_len if avg_len != 0 else 0

        return {
            "average_sentence_length": avg_len,
            "std_dev_sentence_length": std_len,
            "average_change": avg_change,
            "coefficient_of_variation": coeff_var
        }



class StylisticRichnessAggregator:
    def __init__(self, text, api_key, embedding_model, spacy_model):
        """
        text: full text
        api_key: for LiteraryDeviceDetector
        embedding_model: for SemanticDensityAnalyzer
        spacy_model: already loaded nlp instance of spaCy
        """
        self.text = text
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.spacy_nlp = spacy_model

        # 1. Semantic density
        self.semantic_density_analyzer = SemanticDensityAnalyzer(text, embedding_model=embedding_model)
        # 2. Syntactic complexity
        self.syntactic_complexity_analyzer = SyntacticComplexityAnalyzer(text, spacy_model=spacy_model)
        # 3. Literary Resources Detector
        self.literary_device_detector = LiteraryDeviceDetector(api_key=api_key)
        # 4. New: Rhythm analysis
        self.rhythm_analyzer = SentenceRhythmAnalyzer(text, spacy_nlp=spacy_model)

    def analyze_linguistic_variety(self):
        """
        Already existed: it analyses linguistic variety and syntactic complexity.
        We will add a 'basic' TTR calculation here,
        but now we'll use the LexicalDiversityAnalyzer class for more metrics.
        """
        syntactic_metrics = self.syntactic_complexity_analyzer.analyze()
        
        # We tokenise the text into words for the LexicalDiversityAnalyzer.
        tokens = [token.text.lower() for token in self.spacy_nlp(self.text) if token.is_alpha]
        
        lex_div_analyzer = LexicalDiversityAnalyzer(tokens)
        lex_div_metrics = lex_div_analyzer.compute_all()
        
        # We combine syntactic and lexical diversity metrics
        return {
            "lexical_diversity": lex_div_metrics, 
            **syntactic_metrics
        }

    def analyze_rhetorical_devices(self):
        """
        Detects rhetorical devices and calculates their frequency and diversity.
        """
        literary_devices = self.literary_device_detector.detect_literary_devices(self.text)
        if literary_devices is None:
            return {}, {"counts": {}, "total_count": 0, "variety": 0}
        return literary_devices, self.literary_device_detector.quantify_devices(literary_devices)
   
    def analyze_semantic_density(self):
        """
        Calculate semantic density with word-level embeddings
        """
        return self.semantic_density_analyzer.analyze()

    def analyze_sentence_rhythm(self):
        """
        Measures the variation/rhythm of sentences.
        """
        return self.rhythm_analyzer.analyze_rhythm()

    def compute_stylistic_richness(self):
        """
        Combines all metrics into a unified report.
        """
        linguistic_metrics = self.analyze_linguistic_variety()
        rhetorical_devices, rhetorical_devices_count = self.analyze_rhetorical_devices()
        semantic_density = self.analyze_semantic_density()
        rhythm_metrics = self.analyze_sentence_rhythm()

        return {
            "linguistic_metrics": linguistic_metrics,
            "rhetorical_devices": rhetorical_devices_count, 
            "rhetorical_devices_cats": rhetorical_devices,
            "semantic_density": semantic_density,
            "sentence_rhythm": rhythm_metrics
        }
