# When AI Writes Like Humans

This repository contains the code for reproducibility of the experiments described in the paper:

**"When AI Writes Like Humans: Capturing the Emergent Patterns of Literary Judgment via Intrinsic Textual Metrics"**

The code implements all main analyses from the paper, including dataset preparation, computation of intrinsic textual metrics, and modeling reader preferences.

## Repository Structure

- **Data Preparation & Loading**  
  - `0_1_datasets_preparation/` – Scripts to prepare and clean the used datasets from the original format.  

- **Datasets** 
    - `datasets` - Datasets already in both ordinal ranking and pairwise ranking for all subsequent analyses. It contains the result of `0_1_datasets_preparation/`.

- **Text Metrics Computation**  
  - `0_2_compute_text_metrics/` – Code to calculate metrics such as coherence, originality, sentiment, and stylistic richness.  
  - [`1_1_selecting_metrics.ipynb`](1_1_selecting_metrics.ipynb) – Notebook showing how key features were selected.

- **Descriptive Analysis of Texts Features**  
  - [`1_0_describing_human_vs_AI_texts.ipynb`](1_0_describing_human_vs_AI_texts.ipynb) – Describes differences between AI- and human-authored texts.  
  - [`1_2_paradigmatic_text_by_reader.ipynb`](1_2_paradigmatic_text_by_reader.ipynb) – Computes reader-specific "ideal text" representations.
  - [`1_3_selected_metrics_correlation.ipynb`](1_3_selected_metrics_correlation.ipynb) – Analyzes correlations among textual metrics.
  - [`1_4_cluster_texts_by_metrics_vector.ipynb`](1_4_cluster_texts_by_metrics_vector.ipynb) – Clusters texts based on their features.

- **Preference Modeling & Visualization**  
  - [`2_1_train_models.py`](2_1_train_models.py) and [`trainer.py`](trainer.py) – Scripts to train preference models (e.g., Random Forest) that capture reader-specific weighting of textual features.
  - [`2_2_model_performance_box_plot.ipynb`](2_2_model_performance_box_plot.ipynb) – Evaluates model performances.
  - [`3_1_readers_clusters.ipynb`](3_1_readers_clusters.ipynb) and [`3_2_readers_mean_clusters_radar_chart.ipynb`](3_2_readers_mean_clusters_radar_chart.ipynb) – Visualize clustering of readers and compare feature importance profiles.

## Paper Correspondence

- **Textual Separability (RQ1):**  
  Notebooks such as [`1_0_describing_human_vs_AI_texts.ipynb`](1_0_describing_human_vs_AI_texts.ipynb) and [`1_4_cluster_texts_by_metrics_vector.ipynb`](1_4_cluster_texts_by_metrics_vector.ipynb) generate PCA projections and clustering analyses featured in the paper.

- **Reader Preferences (RQ2, RQ3 and RQ4):**  
  The preference modeling using additive utility (with features weighted by reviewer importance) is implemented in [`2_1_train_models.py`](2_1_train_models.py) and further analyzed in [`3_1_readers_clusters.ipynb`](3_1_readers_clusters.ipynb) and [`3_2_readers_mean_clusters_radar_chart.ipynb`](3_2_readers_mean_clusters_radar_chart.ipynb).


## How to Reproduce

1. **Dataset Preparation:**  
   Run the scripts in `0_1_datasets_preparation/` for derivate all the rankings from de original datasets.

2. **Text Metric Computation:**  
   Set your OpenAI API in `text_metric_evaluator.py` and then execute `main.py` in `0_2_compute_text_metrics/`. Sentiment and emotion analysis models require a GPU. It took about 12 hours in a A5000 24GB. The results of these metric computations can be found at [`datasets/0_metrics`](datasets/0_metrics).

3. **Analysis & Modeling:**  
   Run the notebooks in sequence starting with [`1_0_describing_human_vs_AI_texts.ipynb`](1_0_describing_human_vs_AI_texts.ipynb), then train preference models for each reader with [`2_1_train_models.py`](2_1_train_models.py). It took around 30 hours in a i7-10700 CPU.

4. **Visualization:**  
   At the end of the sequential execution of the notebooks the results will be found in the folders `outputs`, `figures` and `model_results`. In these folders we provide the execution results that we report in the paper.

## Dependencies

- Python 3.x  
- Jupyter Notebook  
- Libraries listed in `requirements.txt`

## Datasets Attribution

  For each analyzed dataset, please note the following corresponding paper attributions:
  - **SLM Dataset:**: Marco, G., Rello, L., & Gonzalo, J. (2025, January). *Small language models can outperform humans in short creative writing: A study comparing SLMs with humans and LLMs*. In O. Rambow, L. Wanner, M. Apidianaki, H. Al-Khalifa, & S. Schockaert (Eds.), Proceedings of the 31st International Conference on Computational Linguistics (pp. 6552–6570). Association for Computational Linguistics. Retrieved January 29, 2025, from https://aclanthology.org/2025.coling-main.437/
  - **HANNA Dataset:** Chhun, C., Colombo, P., Suchanek, F. M., & Clavel, C. (2022, October). *Of human criteria and automatic metrics: A benchmark of the evaluation of story generation*. In Proceedings of the 29th International Conference on Computational Linguistics (pp. 5794–5836). International Committee on Computational Linguistics. Retrieved January 08, 2025, from https://aclanthology.org/2022.coling-1.509/

  - **Confederacy Dataset:** Gómez-Rodríguez, C., & Williams, P. (2023, December). *A confederacy of models: A comprehensive evaluation of LLMs on creative writing*. In H. Bouamor, J. Pino, & K. Bali (Eds.), Findings of the Association for Computational Linguistics: EMNLP 2023 (pp. 14504–14528). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.findings-emnlp.966

  - **TTCW Dataset:** Chakrabarty, T., Laban, P., Agarwal, D., Muresan, S., & Wu, C.-S. (2024, May). *Art or artifice? Large language models and the false promise of creativity*. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems (pp. 1–34). Association for Computing Machinery. https://doi.org/10.1145/3613904.3642731

  - **Pron vs Prompt Dataset:** Marco, G., Gonzalo, J., Mateo-Girona, M. T., & Santos, R. D. C. (2024, November). *Pron vs prompt: Can large language models already challenge a world-class fiction author at creative text writing?* In Y. Al-Onaizan, M. Bansal, & Y.-N. Chen (Eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (pp. 19654–19670). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.emnlp-main.1096





