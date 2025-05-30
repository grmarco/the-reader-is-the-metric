import math
import statistics
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
import spacy
from spacy.cli import download as spacy_download

# Download resources from NLTK (first time only)
# Uncomment the following lines if you have not previously downloaded the resources
# nltk.download('stopwords')
# nltk.download('wordnet')

class TextProcessor:
    def __init__(self, text, embedding_model, spacy_model, words_per_segment=50, use_lemmatization=True):
        """
        Initialises the word processor.

        :param text: Text to process.
        :param embedding_model: Embedding model.
        :param spacy_model: SpaCy model loaded for lemmatisation.
        :param words_per_segment: Number of words per segment.
        :param use_lemmatization: Boolean to enable/disable lemmatization.
        """
        self.text = text
        self.words_per_segment = words_per_segment
        self.use_lemmatization = use_lemmatization
        self.model = embedding_model
        self.nlp = spacy_model
        self.segments = self.tokenize()

    def preprocess_text(self, text):
        """
        Preprocesses text by removing punctuation, numbers, special characters,
        converting to lowercase, removing stopwords and applying lemmatisation if enabled.

        :param text: Original text.
        return: Preprocessed text.
        """
        # Eliminar caracteres no alfabéticos
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convertir a minúsculas
        text = text.lower()
        # Tokenizar
        words = text.split()
        # Eliminar stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        if self.use_lemmatization:
            try:
                # Unir palabras para procesar con SpaCy
                text = ' '.join(words)
                doc = self.nlp(text)
                words = [token.lemma_ for token in doc]
            except Exception as e:
                print(f"Error during lemmatisation: {e}")
                # Mantener las palabras sin lematizar si ocurre un error

        return ' '.join(words)

    def tokenize(self):
        """
        Pre-processes and divides text into segments of a specified number of words.
        """
        preprocessed_text = self.preprocess_text(self.text)
        words = preprocessed_text.split()
        segments = []
        for i in range(0, len(words), self.words_per_segment):
            segment = ' '.join(words[i:i + self.words_per_segment])
            segments.append(segment)
        return segments

    def compute_embeddings(self):
        """
        Calculates embeddings for text segments.
        """
        return self.model.encode(self.segments, show_progress_bar=True)

class ThemeClustering:
    def __init__(self, embeddings, distance_threshold=0.3, linkage='average', pca_components=50):
        """
        Initialises thematic clustering.

        :param embeddings: Segment embeddings.
        :param distance_threshold: Distance threshold for AgglomerativeClustering.
        :param linkage: Linkage method for AgglomerativeClustering.
        :param pca_components: Number of components for PCA.
        """
        self.embeddings = embeddings
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.pca_components = pca_components
        self.cluster_labels = self.cluster_embeddings()

    def cluster_embeddings(self):
        """
        Reduces the dimensionality of embeddings and groups using hierarchical clustering.
        """
        n_samples, n_features = self.embeddings.shape

        # Determine the number of PCA components
        n_components = min(self.pca_components, n_samples, n_features)
        if n_components < 1:
            raise ValueError("El número de componentes para PCA debe ser al menos 1.")

        # Dimensionality reduction with PCA
        reduced_embeddings = self.embeddings

        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="cosine",
            linkage=self.linkage
        )
        labels = clustering_model.fit_predict(reduced_embeddings)
        
        return labels

class ThematicMetrics:
    def __init__(self, segments, embeddings, cluster_labels):
        """
        Initialise thematic metrics.

        :param segments: Text segments.
        :param embeddings: Segment embeddings.
        :param cluster_labels: Cluster labels.
        """
        self.segments = segments
        self.embeddings = embeddings
        self.cluster_labels = cluster_labels

    def calculate_entropy(self):
        """
        Calculates the normalised entropy of the clusters.
        Includes all clusters, even those with only one segment.
        """
        labels = self.cluster_labels[self.cluster_labels >= 0]  
        if len(labels) == 0:
            return 0.0

        counts = np.bincount(labels)
        if len(counts) == 0:
            return 0.0

        probs = counts / counts.sum()
        ent = entropy(probs)
        return ent / np.log(len(probs)) if len(probs) > 1 else 0.0

    def calculate_lexical_diversity(self):
        """
        Calculate lexical diversity per cluster.
        """
        cluster_words = defaultdict(set)
        for label, segment in zip(self.cluster_labels, self.segments):
            if label >= 0:
                words = set(segment.lower().split())
                cluster_words[label].update(words)
        total_unique_words = set(word for words in cluster_words.values() for word in words)
        return len(total_unique_words) / len(self.segments) if len(self.segments) > 0 else 0.0

    def calculate_inter_theme_similarity(self):
        """
        Calculates the average similarity between topics (clusters).
        """
        unique_labels = np.unique(self.cluster_labels)
        cluster_embeddings = [
            np.mean(self.embeddings[np.where(self.cluster_labels == label)], axis=0)
            for label in unique_labels if label != -1
        ]
        if len(cluster_embeddings) < 2:
            return 1.0
        similarities = cosine_similarity(cluster_embeddings)
        # Excluir la diagonal
        off_diagonal = similarities[~np.eye(similarities.shape[0], dtype=bool)].reshape(similarities.shape[0], -1)
        return np.mean(off_diagonal) if off_diagonal.size > 0 else 1.0

    def calculate_exploration_depth(self):
        """
        Calculates the average size and standard deviation of the clusters.
        """
        cluster_sizes = [np.sum(self.cluster_labels == label) for label in np.unique(self.cluster_labels) if label >= 0]
        return np.mean(cluster_sizes), np.std(cluster_sizes)   

    def compute_metrics(self):
        """
        Calculates all relevant metrics: entropy, similarity between topics and exploration.
        """
        entropy_score = self.calculate_entropy()
        inter_theme_similarity = self.calculate_inter_theme_similarity()
        mean_cluster_size, std_cluster_size = self.calculate_exploration_depth()
        return {
            "entropy_score": entropy_score,
            "inter_theme_similarity": inter_theme_similarity,
            "mean_cluster_size": mean_cluster_size,
            "std_cluster_size": std_cluster_size,
        }

class SemanticGraph:
    def __init__(self, cluster_embeddings, similarity_threshold=0.3):
        """
        Initialises the semantic network.

        :param cluster_embeddings: Average embeddings for each cluster.
        :param similarity_threshold: Similarity threshold for creating connections between clusters.
        """
        self.cluster_embeddings = cluster_embeddings
        self.similarity_threshold = similarity_threshold

    def compute_graph_metrics(self):
        """
        Constructs a graph based on the similarity between clusters and calculates relevant metrics.
        """
        G = self.build_graph()
        graph_density = self.compute_density(G)
        modularity = self.compute_modularity(G)
        return {
            "graph_density": graph_density,
            "graph_modularity": modularity,
            "graph": G,  
        }

    def build_graph(self):
        """
        Constructs a graph based on the similarity between thematic clusters.
        """
        similarities = cosine_similarity(self.cluster_embeddings)
        G = nx.Graph()
        for i in range(len(self.cluster_embeddings)):
            G.add_node(f"Cluster {i}")
            for j in range(i + 1, len(self.cluster_embeddings)):
                if similarities[i, j] > self.similarity_threshold:
                    G.add_edge(f"Cluster {i}", f"Cluster {j}", weight=similarities[i, j])
        return G

    def compute_density(self, G):
        """
        Calculates the density of the graph.
        """
        return nx.density(G)

    def compute_modularity(self, G):
        """
        Calculates the modularity of the network.
        """
        if len(G) == 0:
            return 0
        try:
            communities = nx.community.greedy_modularity_communities(G)
            return nx.community.modularity(G, communities)
        except Exception as e:
            print(f"Error in calculating modularity: {e}")
            return 0

    def visualize_graph(self, G, title="Semantic Graph"):
        """
        Display the semantic network.
        """
        if len(G.nodes) == 0:
            print("Empty network. No nodes to display.")
            return
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(8,6))
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=12)
        plt.title(title)
        plt.axis('off')
        plt.show()

class ThematicAnalysis:
    def __init__(self, text, embedding_model, spacy_model, words_per_segment=50, use_lemmatization=True, clustering_threshold=0.3, graph_similarity_threshold=0.3, linkage='average'):
        """
        Initialises the thematic analysis.

        :param text: Text to parse.
        :param embedding_model: Embedding model.
        :param spacy_model: SpaCy model loaded for lemmatisation.
        :param words_per_segment: Number of words per segment.
        :param use_lemmatization: Boolean to enable/disable lemmatization.
        param clustering_threshold: Distance threshold for clustering.
        :param graph_similarity_threshold: Similarity threshold for the graph.
        param linkage: Linkage method for clustering.
        """

        self.text_processor = TextProcessor(text, embedding_model, spacy_model, words_per_segment, use_lemmatization)
        self.embeddings = self.text_processor.compute_embeddings()
        
        # Check that there are enough segments for clustering.
        if len(self.text_processor.segments) < 2:
            print("There are not enough segments to apply clustering.")
            self.cluster_labels = np.array([])
        else:
            self.theme_clustering = ThemeClustering(
                self.embeddings,
                distance_threshold=clustering_threshold,
                linkage=linkage
            )
            self.thematic_metrics = ThematicMetrics(
                self.text_processor.segments,
                self.embeddings,
                self.theme_clustering.cluster_labels
            )
            self.cluster_embeddings = [
                np.mean(self.embeddings[np.where(self.theme_clustering.cluster_labels == label)], axis=0)
                for label in np.unique(self.theme_clustering.cluster_labels) if label != -1
            ]
            self.semantic_graph = SemanticGraph(
                self.cluster_embeddings,
                similarity_threshold=graph_similarity_threshold  # Pasar el umbral aquí
            )

    def compute_thematic_depth(self, visualize=False):
        """
        Calculates the thematic depth by combining metrics of the clusters and the graph.

        :param visualize: Boolean to enable/disable the visualisation of the network.
        return: Dictionary with the calculated metrics.
        """
        if len(self.text_processor.segments) < 2:
            print("Metrics cannot be calculated without at least 2 segments.")
            return {
                "entropy_score": 0.0,
                "inter_theme_similarity": 0.0,
                "graph_density": 0.0,
                "graph_modularity": 0.0,
            }
        
        metrics = self.thematic_metrics.compute_metrics()
        graph_metrics = self.semantic_graph.compute_graph_metrics()

        if visualize:
            self.semantic_graph.visualize_graph(graph_metrics["graph"], title="Semantic Graph")

        return {
            "entropy_score": metrics["entropy_score"],
            "inter_theme_similarity": metrics["inter_theme_similarity"],
            "graph_density": graph_metrics["graph_density"],
            "graph_modularity": graph_metrics["graph_modularity"],
        }

    def visualize_clusters(self):
        """
        Visualise clusters using PCA to reduce dimensionality.
        """
        if len(self.text_processor.segments) < 2:
            print("There are not enough segments to display clusters.")
            return
        
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(self.embeddings)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=reduced_embeddings[:,0], 
            y=reduced_embeddings[:,1], 
            hue=self.theme_clustering.cluster_labels, 
            palette='viridis',
            legend='full'
        )
        plt.title("Visualisation of Thematic Clusters")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
