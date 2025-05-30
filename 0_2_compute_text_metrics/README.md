# Stylistic Analysis

This module performs an in-depth stylistic analysis of text. It provides classes that analyze various linguistic aspects such as literary devices, syntactic complexity, semantic density, lexical diversity, and sentence rhythm. The main classes included in this file are described below.

## 1. LiteraryDeviceDetector

- **Purpose:**  
  Detects literary devices (e.g., metaphors, similes, alliterations, etc.) in a given text by leveraging an OpenAI model. It expects the text to be analyzed and returns a structured JSON object containing lists of detected instances for each literary category.

- **Key Methods:**  
  - `__init__(self, api_key)`:  
    Initializes the detector with the OpenAI API key and creates a client instance.
    
  - `generate_response(self, prompt, model="gpt-4o", temperature=0.2)`:  
    Sends a prompt to the OpenAI API and retrieves the response.
    
  - `detect_literary_devices(self, text)`:  
    Replaces the placeholder in a superprompt with the provided text, calls the model, and attempts to return the detected literary devices as a JSON.
    
  - `quantify_devices(self, literary_devices)`:  
    Counts the number and variety of literary devices detected.

## 2. SyntacticComplexityAnalyzer

- **Purpose:**  
  Analyzes syntactic features of a text using spaCy. It assesses the complexity of sentence structures through various measures.

- **Key Methods:**  
  - `__init__(self, text, spacy_model="en_core_web_sm")`:  
    Sets up the analyzer by loading a spaCy language model.
    
  - `average_sentence_length(self)`:  
    Computes the average token count per sentence.
    
  - `count_subordinate_clauses(self)`:  
    Counts the subordinate clauses indicated by specific dependency labels (e.g., "mark", "advcl").
    
  - `dependency_variety(self)`:  
    Determines the number of unique dependency relations in the text.
    
  - `sentence_complexity_index(self)`:  
    Computes an index of syntactic complexity based on subordinate clause frequency.
    
  - `average_tree_depth(self)`:  
    Calculates the average depth of dependency trees across tokens.
    
  - `measure_nested_subordination(self)`:  
    Assesses how deeply subordinate clauses are nested within sentences.

## 3. SemanticDensityAnalyzer

- **Purpose:**  
  Evaluates the semantic density of the text using word embeddings. It calculates the similarity between consecutive word embeddings and the entropy of word frequencies.

- **Key Methods:**  
  - `__init__(self, text, embedding_model)`:  
    Initializes the analyzer, tokenizes the text, and sets the embedding model.
    
  - `compute_word_embeddings(self)`:  
    Generates embeddings for each token.
    
  - `calculate_cosine_similarity(self)`:  
    Computes the average cosine similarity between adjacent word embeddings.
    
  - `calculate_entropy(self)`:  
    Calculates the entropy of word frequencies to gauge text diversity.
    
  - `analyze(self)`:  
    Returns a summary dictionary with the computed semantic density metrics.

## 4. LexicalDiversityAnalyzer

- **Purpose:**  
  Computes multiple metrics of lexical diversity such as Type-Token Ratio (TTR), log-TTR, and MTLD (Measure of Textual Lexical Diversity).

- **Key Methods:**  
  - `ttr(self)`:  
    Calculates the type-to-token ratio.
    
  - `log_ttr(self)`:  
    Computes the logarithm of TTR.
    
  - `mtld(self, ttr_threshold=0.72)`:  
    Approximates the MTLD using both forward and reverse calculations.
    
  - `compute_all(self)`:  
    Aggregates all lexical diversity metrics into a structured dictionary.

## 5. SentenceRhythmAnalyzer

- **Purpose:**  
  Measures the rhythm and variation in sentence length throughout the text. This can correlate with the readability and stylistic flow of a document.

- **Key Methods:**  
  - `__init__(self, text, spacy_nlp)`:  
    Initializes the analyzer by parsing the input text into sentences using spaCy.
    
  - `get_sentence_lengths(self)`:  
    Returns a list of token counts for each sentence.
    
  - `analyze_rhythm(self)`:  
    Calculates metrics such as average sentence length, standard deviation, average change between sentences, and coefficient of variation.

## 6. StylisticRichnessAggregator

- **Purpose:**  
  Acts as a central aggregator that combines all the individual analyses into a comprehensive report of the text’s stylistic richness. It brings together metrics from semantic, syntactic, lexical, and rhetorical analyses.

- **Key Methods:**  
  - `__init__(self, text, api_key, embedding_model, spacy_model)`:  
    Initializes the aggregator with the required API key and models for stylistic analyses.
    
  - `analyze_linguistic_variety(self)`:  
    Computes both syntactic metrics (via `SyntacticComplexityAnalyzer`) and lexical diversity (via `LexicalDiversityAnalyzer`).
    
  - `analyze_rhetorical_devices(self)`:  
    Uses `LiteraryDeviceDetector` to obtain and quantify the rhetorical/literary devices present in the text.
    
  - `analyze_semantic_density(self)`:  
    Analyzes the semantic density using the `SemanticDensityAnalyzer`.
    
  - `analyze_sentence_rhythm(self)`:  
    Retrieves measures of sentence rhythm using the `SentenceRhythmAnalyzer`.
    
  - `compute_stylistic_richness(self)`:  
    Aggregates all the metrics from the various analyses into one unified report.

## Usage Example

To use the stylistic richness analysis, initialize a `StylisticRichnessAggregator` with the text, an OpenAI API key, an embedding model, and a loaded spaCy model. Then call the `compute_stylistic_richness` method. For example:

```python
from style import StylisticRichnessAggregator
import spacy
# Assume embedding_model is already defined (e.g., from SentenceTransformers)

nlp = spacy.load("en_core_web_sm")
text = "Your sample text goes here..."
api_key = "YOUR_OPENAI_API_KEY"

aggregator = StylisticRichnessAggregator(text, api_key, embedding_model, nlp)
result = aggregator.compute_stylistic_richness()
print(result)
```

This will output a comprehensive report containing:
- Linguistic metrics (syntactic complexity and lexical diversity)
- Rhetorical device counts and types
- Semantic density measurements
- Sentence rhythm details

---

This module combines advanced Natural Language Processing techniques with statistical analysis to provide a detailed understanding of a text’s stylistic properties.