import json
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import spacy
from text_metric_evaluator import TextMetricEvaluator
import numpy as np
import os

def convert_to_native_types(data):
    if isinstance(data, dict):
        return {k: convert_to_native_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(v) for v in data]
    elif isinstance(data, np.generic):
        return data.item()
    else:
        return data

def load_previous_progress(output_file):
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            return json.load(f)
    return []

def main():
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    lm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    lm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    lm_model.eval()
    spacy_nlp = spacy.load("en_core_web_trf")

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="siebert/sentiment-roberta-large-english",
        device="cuda"
    )
    emotion_pipeline = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,
        device="cuda"
    )

    evaluator = TextMetricEvaluator(
        embedding_model=embedding_model,
        lm_tokenizer=lm_tokenizer,
        lm_model=lm_model,
        spacy_nlp=spacy_nlp,
        sentiment_pipeline=sentiment_pipeline,
        emotion_pipeline=emotion_pipeline
    )
    
    stories_files = [
        "../datasets/0_texts/confederacy_short_stories.json",
        "../datasets/0_texts/slm_short_stories.json",
        "../datasets/0_texts/pronvsprompt_short_stories.json",
        "../datasets/0_texts/ttcw_short_stories.json",
        "../datasets/0_texts/hanna_short_stories.json",
    ]

    total_stories = 0
    for file in stories_files:
        with open(file, 'r') as f:
            stories = json.load(f)
            print(f"Loading stories from {file}")
            total_stories += len(stories)

    processed_stories = 0
    start_time = time.time()

    for file in stories_files:
        with open(file, 'r') as f:
            stories = json.load(f)

        output_file = file.replace(".json", "_metrics.json").replace('texts', 'metrics')
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(output_file)
        metrics_list = load_previous_progress(output_file)
        already_processed_ids = {metric['story_id'] for metric in metrics_list}
        processed_stories += len(already_processed_ids)

        for story in stories:
            if story['story_id'] in already_processed_ids:
                print(f"Skipping already processed story_id: {story['story_id']}")
                continue

            text = story['content']
            print(f"Processing story_id: {story['story_id']}, story_idx: {story['story_idx']}")
            metrics = evaluator.compute_metrics(text, [s['content'] for s in stories])
            metrics = convert_to_native_types(metrics)
            metrics['story_id'] = story['story_id']
            metrics['story_idx'] = story['story_idx']

            metrics_list.append(metrics)
            processed_stories += 1

            elapsed_time = time.time() - start_time
            avg_time_per_story = elapsed_time / processed_stories
            remaining_time = avg_time_per_story * (total_stories - processed_stories)
            remaining_time_hours = remaining_time / 3600

            print(f"Processed {processed_stories}/{total_stories} stories.")
            print(f"Elapsed time: {elapsed_time:.2f} seconds.")
            print(f"Estimated remaining time: {remaining_time_hours:.2f} hours.")

            with open(output_file, 'w') as out_f:
                json.dump(metrics_list, out_f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()