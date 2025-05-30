import csv
import json
import os

# Define the paths
csv_path = 'input/anonymization/mapping-grouped.csv'
corpus_dir = 'input/corpus-full'
output_json = 'output.json'

# Read the CSV file
mapping = []
with open(csv_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip header
    for row in csv_reader:
        original, anonymized = row
        mapping.append((original, anonymized))

# Read the text files and create the JSON structure
data = []
story_idx = 0
for original, anonymized in mapping:
    original_path = os.path.join(corpus_dir, original)
    if os.path.exists(original_path):
        try:
            with open(original_path, mode='r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                story_idx += 1
                data.append({
                    "story_idx": story_idx,
                    "story_id": anonymized.replace('.txt', ''),
                    "story_id_original": original.replace('.txt', ''),
                    "content": content
                })
        except Exception as e:
            print(f"Error reading {original_path}: {e}")
    else:
        print(f"Warning: {original_path} does not exist.")

# Write the JSON file
with open(output_json, mode='w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f"JSON file created: {output_json}")