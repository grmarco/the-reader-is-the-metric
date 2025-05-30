from trainer import LiteraryPreferenceAnalyzer
import pandas as pd
from config import selected_features

# List of datasets
datasets = ['confederacy',  'ttcw', 'pronvsprompt', 'slm', 'hanna']
# Path templates for metrics and pairs data
metrics_path_template = 'datasets/1_metrics/{}_short_stories_metrics.json'

trainer_by_reader = LiteraryPreferenceAnalyzer(hyperparameter_search=True, results_dir='model_results/by_reader')
pairs_path_template = 'datasets/1_1_pairwise_preferences/by_user/{}_pairs_balanced.json'
# Loop through datasets and analyze each one
for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    metrics_path = metrics_path_template.format(dataset)
    pairs_path = pairs_path_template.format(dataset)

    # Load the datasets
    trainer_by_reader.load_metrics_data(metrics_path)
    trainer_by_reader.load_pairs_data(pairs_path)

    # training model for each reader
    results = trainer_by_reader.run_training(selected_columns=selected_features)

trainer_by_reader = LiteraryPreferenceAnalyzer(hyperparameter_search=True, results_dir='model_results/baseline')
pairs_path_template = 'datasets/1_1_pairwise_preferences/baseline/{}_pairs_balanced.json'
# Loop through datasets and analyze each one
for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    metrics_path = metrics_path_template.format(dataset)
    pairs_path = pairs_path_template.format(dataset)

    # Load the datasets
    trainer_by_reader.load_metrics_data(metrics_path)
    trainer_by_reader.load_pairs_data(pairs_path)

    # Answer research questions using the selected features
    results = trainer_by_reader.run_training(selected_columns=selected_features)


