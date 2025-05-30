import json
import pandas as pd

# Cargar el archivo de mapeo
mapping_df = pd.read_csv('/data/gmarco/canon_explicability/datasets/confederacy/input/github_original/anonymization/mapping-grouped.csv')

# Eliminar la extensión .txt de las columnas 'Original' y 'Anonymized'
mapping_df['Original'] = mapping_df['Original'].str.replace('.txt', '')
mapping_df['Anonymized'] = mapping_df['Anonymized'].str.replace('.txt', '')

# Crear un diccionario de mapeo
mapping_dict = dict(zip(mapping_df['Anonymized'], mapping_df['Original']))

# Función para actualizar story_idx
def update_story_idx(story):
    anonymized_id = story['story_id']
    original_id = mapping_dict.get(anonymized_id, '')
    if original_id:
        story['story_idx'] = f"{anonymized_id}_{original_id}"
    return story

# Función para actualizar preferred_text y other_text
def update_pair(pair):
    preferred_id = pair['preferred_text']
    other_id = pair['other_text']
    preferred_original = mapping_dict.get(preferred_id, '')
    other_original = mapping_dict.get(other_id, '')
    if preferred_original:
        pair['preferred_text'] = f"{preferred_id}_{preferred_original}"
    if other_original:
        pair['other_text'] = f"{other_id}_{other_original}"
    return pair

# Cargar y actualizar confederacy_short_stories_metrics.json
with open('../datasets/1_metrics/confederacy_short_stories_metrics.json', 'r') as f:
    metrics_data = json.load(f)

metrics_data = [update_story_idx(story) for story in metrics_data]

with open('../datasets/1_metrics/confederacy_short_stories_metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=4)

# Cargar y actualizar confederacy_pairs_balanced.json
with open('../datasets/1_pairwise/confederacy_pairs_balanced.json', 'r') as f:
    pairs_data = json.load(f)

pairs_data = [update_pair(pair) for pair in pairs_data]

with open('../datasets/1_pairwise/confederacy_pairs_balanced.json', 'w') as f:
    json.dump(pairs_data, f, indent=4)

# Cargar y actualizar confederacy_short_stories.json
with open('../datasets/0_texts/confederacy_short_stories.json', 'r') as f:
    stories_data = json.load(f)

# Actualizar story_id y story_id_original en confederacy_short_stories.json
for story in stories_data:
    anonymized_id = story['story_id']
    original_id = mapping_dict.get(anonymized_id, '')
    if original_id:
        story['story_id'] = f"{anonymized_id}_{original_id}"
        story['story_id_original'] = original_id

with open('../datasets/0_texts/confederacy_short_stories.json', 'w') as f:
    json.dump(stories_data, f, indent=4)