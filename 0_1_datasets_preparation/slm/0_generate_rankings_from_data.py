import pandas as pd

# Cargar el dataset desde la ruta especificada
data = pd.read_csv(r'/home/gmarco/workspace/canon_explicability/datasets/slm/slm_ranking_generation/input/slm_answers.csv')

# Convertir el formato largo a formato ancho para que cada pregunta sea una columna
wide_data = data.pivot_table(
    index=['user', 'title'], 
    columns='question', 
    values='answer'
).reset_index()

# Agregar las columnas writer y experiment al DataFrame
wide_data = wide_data.merge(
    data[['user', 'title', 'writer', 'experiment']].drop_duplicates(), 
    on=['user', 'title'], 
    how='left'
)

# Calcular la media de las respuestas por usuario y título
# Asumiendo que las columnas de respuestas comienzan desde la tercera columna hasta dos antes del final
wide_data['average_score'] = wide_data.iloc[:, 2:-2].mean(axis=1)

# Crear una columna de ranking por usuario basado en el puntaje promedio
wide_data['rank'] = wide_data.groupby('user')['average_score'].rank(method='dense', ascending=False).astype(int)

# Opcional: Ordenar el DataFrame por usuario y ranking
wide_data = wide_data.sort_values(['user', 'rank'])

# Guardar el resultado en un nuevo archivo CSV
output_file_path = r'/home/gmarco/workspace/canon_explicability/datasets/slm/slm_ranking_generation/output/ranked_answers.csv'
wide_data.to_csv(output_file_path, index=False)

print("Ranking completado y datos guardados en 'ranked_answers.csv'.")