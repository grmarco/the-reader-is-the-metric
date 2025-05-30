selected_features = [
    'coherence_analysis.best_num_topics',
    'coherence_analysis.entity_coherence',
    'coherence_analysis.local_coherence_embeddings',
    'coherence_analysis.overall_coherence',
    'originality_analysis.log_likelihood',
    'readability_metrics.smog_index',
    'sentiment_analysis.average_sentiment',
    'sentiment_analysis.emotional_volatility',
    'sentiment_analysis.variance_sentiment',
    'stylistic_analysis.linguistic_metrics.average_sentence_length',
    'stylistic_analysis.linguistic_metrics.average_tree_depth',
    'stylistic_analysis.linguistic_metrics.lexical_diversity.MTLD',
    'stylistic_analysis.linguistic_metrics.max_subordination_depth',
    'stylistic_analysis.rhetorical_devices.variety',
    'stylistic_analysis.sentence_rhythm.std_dev_sentence_length',
    'thematic_analysis.entropy_score',
    'thematic_analysis.graph_density',
]


feature_cols_9 = [
    "coherence_analysis.entity_coherence",
    "coherence_analysis.local_coherence_embeddings",
    "originality_analysis.log_likelihood",
    "readability_metrics.smog_index",
    "sentiment_analysis.average_sentiment",
    "stylistic_analysis.linguistic_metrics.max_subordination_depth",
    "stylistic_analysis.rhetorical_devices.variety",
    "thematic_analysis.graph_density",
    "thematic_analysis.inter_theme_similarity"
]

stories_metrics_filepath = [
    'datasets/1_metrics/confederacy_short_stories_metrics.json',
    'datasets/1_metrics/ttcw_short_stories_metrics.json',
    'datasets/1_metrics/slm_short_stories_metrics.json',
    'datasets/1_metrics/pronvsprompt_short_stories_metrics.json',
    'datasets/1_metrics/hanna_short_stories_metrics.json'
]

rankings_filepath = [
    'datasets/1_2_rankings/confederacy_rankings.csv',    
    'datasets/1_2_rankings/pronvsprompt_rankings.csv',
    'datasets/1_2_rankings/slm_rankings.csv',
    'datasets/1_2_rankings/ttcw_rankings.csv',
    'datasets/1_2_rankings/hanna_rankings.csv',
    ]