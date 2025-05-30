import pandas as pd
from itertools import combinations
import json

# Define the paths
csv_path = 'input/user_ratings.csv'
output_json = 'output/pairs_balanced_user_ratings.json'

# Read the CSV file
df_ratings = pd.read_csv(csv_path, delimiter=';')

# Calcular la puntuación total para cada historia por usuario
df_ratings['total_score'] = df_ratings.iloc[:, 2:-1].sum(axis=1)

# Generar comparaciones por pares
pairwise_data = []
ties_count = 0

for user, group in df_ratings.groupby('user'):
    for (idx1, text1), (idx2, text2) in combinations(group.iterrows(), 2):
        # Comparar puntuaciones
        if text1['total_score'] > text2['total_score']:
            pairwise_data.append({
                'user_id': user,
                'preferred_text': text1['story_id'],
                'other_text': text2['story_id'],
                'label': 1
            })
            # Generar el par opuesto balanceado
            pairwise_data.append({
                'user_id': user,
                'preferred_text': text2['story_id'],
                'other_text': text1['story_id'],
                'label': 0
            })
        elif text1['total_score'] < text2['total_score']:
            pairwise_data.append({
                'user_id': user,
                'preferred_text': text2['story_id'],
                'other_text': text1['story_id'],
                'label': 1
            })
            # Generar el par opuesto balanceado
            pairwise_data.append({
                'user_id': user,
                'preferred_text': text1['story_id'],
                'other_text': text2['story_id'],
                'label': 0
            })
        else:
            ties_count += 1

# Mostrar el número de empates
print(f"Número de empates encontrados: {ties_count}")

# Guardar el resultado en un archivo JSON
with open(output_json, 'w', encoding='utf-8') as json_file:
    json.dump(pairwise_data, json_file, ensure_ascii=False, indent=4)

print(f"JSON file created: {output_json}")