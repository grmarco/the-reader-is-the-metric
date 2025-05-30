import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import logging
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LiteraryPreferenceAnalyzer:
    def __init__(self, cache_dir='feature_cache', hyperparameter_search=True, results_dir='model_results'):
        self.metrics_data = None
        self.pairs_data = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.cache_dir = cache_dir
        self.hyperparameter_search = hyperparameter_search
        self.dataset_prefix = None
        self.results_csv = None
        self.results_dir = results_dir
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def load_metrics_data(self, file_path):
        logging.info(f"Loading metrics data from {file_path}")
        start_time = time.time()
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract dataset prefix from the file name
        self.dataset_prefix = os.path.basename(file_path).split('_')[0]
        self.results_csv = os.path.join(self.results_dir, f'{self.dataset_prefix}_model_results.csv')
        
        # Use pd.json_normalize to flatten the JSON structure
        self.metrics_data = pd.json_normalize(data)
        self.feature_columns = [col for col in self.metrics_data.columns if col != 'story_id']
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        end_time = time.time()
        logging.info(f"Metrics data loaded in {end_time - start_time:.2f} seconds")
        return self.metrics_data
    def load_pairs_data(self, file_path):
        logging.info(f"Loading pairs data from {file_path}")
        start_time = time.time()
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Use pd.json_normalize to flatten the JSON structure
        self.pairs_data = pd.json_normalize(data)
        
        end_time = time.time()
        logging.info(f"Pairs data loaded in {end_time - start_time:.2f} seconds")
        return self.pairs_data

    def split_train_test_by_pair(self, user_pairs, test_size=0.2, random_state=42):
        """
        Divide los pares de un usuario en train y test a nivel de par. Esto garantiza 
        que si un par (por ejemplo, A vs B) aparece en train, no se repita en test.
        """
        # Crear una clave única para cada par sin importar el orden
        user_pairs = user_pairs.copy()
        user_pairs['pair_key'] = user_pairs.apply(
            lambda row: tuple(sorted([row['preferred_text'], row['other_text']])), axis=1
        )
        # Obtener los pares únicos
        unique_pairs = user_pairs['pair_key'].unique()
        # Usar train_test_split para dividir los pares únicos
        from sklearn.model_selection import train_test_split
        train_unique, test_unique = train_test_split(
            unique_pairs, test_size=test_size, random_state=random_state
        )
        # Seleccionar las filas correspondientes a cada conjunto
        train_pairs = user_pairs[user_pairs['pair_key'].isin(train_unique)].copy()
        test_pairs = user_pairs[user_pairs['pair_key'].isin(test_unique)].copy()
        # Eliminar la columna auxiliar
        train_pairs.drop(columns='pair_key', inplace=True)
        test_pairs.drop(columns='pair_key', inplace=True)
        return train_pairs, test_pairs


    

    def create_feature_matrix(self, pairs_data=None, selected_columns=None, cache_name=None):
        if cache_name:
            X, y = self.load_feature_matrix(cache_name)
            if X is not None and y is not None:
                return X, y

        logging.info("Creating feature matrix")
        start_time = time.time()
        if pairs_data is None:
            pairs_data = self.pairs_data

        # If no columns are specified, use all feature columns
        if selected_columns is None:
            selected_columns = self.feature_columns

        X, y = [], []
        total_examples = len(pairs_data)
        for i, (_, row) in enumerate(pairs_data.iterrows()):
            preferred = self.metrics_data[self.metrics_data['story_id'] == row['preferred_text']]
            other = self.metrics_data[self.metrics_data['story_id'] == row['other_text']]

            if preferred.empty or other.empty:
                continue

            diff_features = [preferred[col].values[0] - other[col].values[0] for col in selected_columns]
            X.append(diff_features)
            y.append(row['label'])

            # Log progress every 500 examples
            if (i + 1) % 500 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_example = elapsed_time / (i + 1)
                remaining_examples = total_examples - (i + 1)
                estimated_time_remaining = avg_time_per_example * remaining_examples
                if estimated_time_remaining > 60:
                    estimated_time_remaining = estimated_time_remaining / 60
                    time_unit = "minutes"
                else:
                    time_unit = "seconds"
                logging.info(f"Processed {i + 1} examples. Elapsed time: {elapsed_time:.2f} seconds. Estimated time remaining: {estimated_time_remaining:.2f} {time_unit}")

        end_time = time.time()
        logging.info(f"Feature matrix created in {end_time - start_time:.2f} seconds")

        X, y = np.array(X), np.array(y)
        X = self.scaler.fit_transform(X)  # Normalize the features
        if cache_name:
            self.save_feature_matrix(X, y, cache_name)
        return X, y

    def save_feature_matrix(self, X, y, name):
        dataset_dir = os.path.join(self.cache_dir, f'{self.dataset_prefix}_{name}')
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        np.save(os.path.join(dataset_dir, f'{name}_X.npy'), X)
        np.save(os.path.join(dataset_dir, f'{name}_y.npy'), y)
        logging.info(f"Feature matrix {name} saved in {dataset_dir}.")

    def load_feature_matrix(self, name):
        dataset_dir = os.path.join(self.cache_dir, f'{self.dataset_prefix}_{name}')
        X_path = os.path.join(dataset_dir, f'{name}_X.npy')
        y_path = os.path.join(dataset_dir, f'{name}_y.npy')
        if os.path.exists(X_path) and os.path.exists(y_path):
            X = np.load(X_path)
            y = np.load(y_path)
            logging.info(f"Feature matrix {name} loaded from {dataset_dir}.")
            return X, y
        else:
            logging.warning(f"Feature matrix {name} not found in {dataset_dir}.")
            return None, None

    def train_and_evaluate_models_split(self, X, y, selected_columns, user_id=None, n_splits=3):
        """
        Entrena y evalúa modelos usando validación cruzada (CV).
        Se entrena varios modelos en distintos splits de train y test y se promedian las métricas.
        Se guardan resultados en CSV distintos para Logistic Regression y Random Forest.
        """
        logging.info("Training and evaluating models with cross validation")
        start_time = time.time()

        models = {
            'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
            'Random Forest': RandomForestClassifier(random_state=42)
        }

        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        }
        
        # Definir columnas para resultados, incluyendo las métricas CV
        columns = [
            'user_id', 'model_name', 'accuracy_train', 'precision_train', 'recall_train', 'f1_train', 
            'accuracy_test', 'precision_test', 'recall_test', 'f1_test', 'top_10_features'
        ] + selected_columns
        
        results = {}
        best_models = {}
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for name, model in models.items():
            metric_accum = {
                'accuracy_train': [],
                'precision_train': [],
                'recall_train': [],
                'f1_train': [],
                'accuracy_test': [],
                'precision_test': [],
                'recall_test': [],
                'f1_test': []
            }
            best_params_list = []
            
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                if self.hyperparameter_search:
                    n_splits_inner = min(3, np.min(np.bincount(y_train)))
                    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=n_splits_inner, n_jobs=-1, verbose=0)
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    best_params_list.append(grid_search.best_params_)
                else:
                    best_model = model
                    best_model.fit(X_train, y_train)
                    best_params_list.append(best_model.get_params())
                    
                preds_train = best_model.predict(X_train)
                preds_test = best_model.predict(X_test)
                
                metric_accum['accuracy_train'].append(accuracy_score(y_train, preds_train))
                metric_accum['precision_train'].append(precision_score(y_train, preds_train))
                metric_accum['recall_train'].append(recall_score(y_train, preds_train))
                metric_accum['f1_train'].append(f1_score(y_train, preds_train))
                metric_accum['accuracy_test'].append(accuracy_score(y_test, preds_test))
                metric_accum['precision_test'].append(precision_score(y_test, preds_test))
                metric_accum['recall_test'].append(recall_score(y_test, preds_test))
                metric_accum['f1_test'].append(f1_score(y_test, preds_test))
            
            results[name] = {
                'accuracy_train': np.mean(metric_accum['accuracy_train']),
                'precision_train': np.mean(metric_accum['precision_train']),
                'recall_train': np.mean(metric_accum['recall_train']),
                'f1_train': np.mean(metric_accum['f1_train']),
                'accuracy_test': np.mean(metric_accum['accuracy_test']),
                'precision_test': np.mean(metric_accum['precision_test']),
                'recall_test': np.mean(metric_accum['recall_test']),
                'f1_test': np.mean(metric_accum['f1_test']),
                'feature_importances': self.get_feature_importance(best_model)
            }
            best_models[name] = best_model
            
            feature_importances = results[name]['feature_importances']
            top_10_features = []
            if feature_importances is not None:
                top_10_features = [selected_columns[i] for i in np.argsort(feature_importances)[-10:][::-1]]
            
            result_row = {
                'user_id': user_id,
                'model_name': name,
                'accuracy_train': results[name]['accuracy_train'],
                'precision_train': results[name]['precision_train'],
                'recall_train': results[name]['recall_train'],
                'f1_train': results[name]['f1_train'],
                'accuracy_test': results[name]['accuracy_test'],
                'precision_test': results[name]['precision_test'],
                'recall_test': results[name]['recall_test'],
                'f1_test': results[name]['f1_test'],
                'top_10_features': ', '.join(top_10_features)
            }
            for feature, importance in zip(selected_columns, 
                                           feature_importances if feature_importances is not None else []):
                result_row[feature] = importance

            results_df = pd.DataFrame([result_row])
            # Definir el path del CSV según el modelo
            if name == 'Logistic Regression':
                csv_dir = os.path.join(self.results_dir, 'lr')
            else:
                csv_dir = os.path.join(self.results_dir, 'rf')
            os.makedirs(csv_dir, exist_ok=True)
            results_csv_path = os.path.join(csv_dir, f'{self.dataset_prefix}_model_results.csv')
            
            # Si el archivo no existe, crearlo con encabezados
            if not os.path.exists(results_csv_path):
                pd.DataFrame(columns=columns).to_csv(results_csv_path, index=False)
            results_df = results_df[columns]
            results_df.to_csv(results_csv_path, mode='a', header=False, index=False)
        end_time = time.time()
        logging.info(f"Models trained and evaluated with CV in {end_time - start_time:.2f} seconds")
        return results, best_models

    def run_training(self, selected_columns=None):
        logging.info("Answering research questions")
        start_time = time.time()
        
        # Solo se realiza análisis por usuario
        per_user_results, user_best_models = self.analyze_per_user(selected_columns)

        # Ordenar resultados por accuracy para el modelo Random Forest, por ejemplo
        sorted_per_user_results = dict(sorted(
            (item for item in per_user_results.items() if 'accuracy' in item[1]['Random Forest']),
            key=lambda item: item[1]['Random Forest']['accuracy'], reverse=True))

        end_time = time.time()
        logging.info(f"Research questions answered in {end_time - start_time:.2f} seconds")
        return {
            "RQ1_Per_User_Results": sorted_per_user_results,
            "User_Best_Models": user_best_models,
        }

    def get_feature_importance(self, model):
        if isinstance(model, LogisticRegression):
            importance = np.abs(model.coef_[0])
        elif isinstance(model, RandomForestClassifier):
            importance = model.feature_importances_
        else:
            importance = None
        return importance

    def analyze_per_user(self, selected_columns=None):
        logging.info("Analyzing per user")
        start_time = time.time()
        user_results = {}
        user_best_models = {}
        user_ids = self.pairs_data['user_id'].unique()
        total_users = len(user_ids)

        for i, user_id in enumerate(user_ids):
            user_pairs = self.pairs_data[self.pairs_data['user_id'] == user_id]
            if user_pairs.empty:
                logging.warning(f"No data for user {user_id}, skipping.")
                continue

            # Crear la matriz de características para todo el usuario
            X, y = self.create_feature_matrix(user_pairs, selected_columns, cache_name=f'user_{user_id}')
            if len(X) < 2:
                logging.warning(f"Not enough data for user {user_id}, skipping.")
                continue

            # Entrenar modelos usando CV sobre el conjunto entero del usuario
            results, best_models = self.train_and_evaluate_models_split(X, y, selected_columns, user_id=user_id)
            user_results[user_id] = results
            user_best_models[user_id] = best_models

            elapsed_time = time.time() - start_time
            avg_time_per_user = elapsed_time / (i + 1)
            remaining_users = total_users - (i + 1)
            estimated_time_remaining = avg_time_per_user * remaining_users
            logging.info(f"Processed {i + 1}/{total_users} users. Estimated time remaining: {estimated_time_remaining:.2f} seconds")

        end_time = time.time()
        logging.info(f"Per user analysis completed in {end_time - start_time:.2f} seconds")
        return user_results, user_best_models
