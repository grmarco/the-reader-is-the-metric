from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
import os
import json
import pandas as pd
import config


# we adapted some of the writers' names to show in the clustering figures
def extract_author(story_id):
    parts = story_id.split('_')
    if parts[0] == 'confederacy' and parts[2].find('human')==-1:
        model = parts[2].split('-')[0]+'_confederacy'
        if model == 'chatgpt':
            return parts[2].split('-')[1]+'_confederacy'
        else:
            return model
    elif parts[0] == 'confederacy' and parts[2].find('human')>-1:
        return 'Human_confederacy'
    elif parts[0] == 'slm' and parts[2] == 'bot':
        return 'BART-large_slm'
    elif parts[0] == 'slm' and parts[2] == 'human':
        return 'Human_slm'
    elif parts[0] == 'pronvsprompt' and parts[2] == 'patricio':
        return 'Human_pronvsprompt'
    elif parts[0] == 'pronvsprompt' and parts[2] == 'gpt4':
        return 'GPT4_pronvsprompt'
    elif parts[0] == 'poetry' and parts[2] == 'real':
        return 'Human_poetry'
    elif parts[0] == 'poetry' and parts[2] == 'AI':
        return 'GPT3.5_poetry'
    elif parts[0] == 'ttcw' and parts[2] == 'GPT3.5':
        return 'GPT3.5_ttcw'
    elif parts[0] == 'ttcw' and parts[2] == 'GPT4':
        return 'GPT4_ttcw'
    elif parts[0] == 'ttcw' and parts[2] == 'NewYorker':
        return 'Human_ttcw'
    elif parts[0] == 'ttcw' and parts[2] == 'Claude':
        return 'Claude_ttcw'
    else:
        return parts[2]

def load_json_to_df(filepaths):
    data = []
    for filepath in filepaths:
        with open(filepath, 'r') as file:
            prefix = os.path.basename(filepath).split('_')[0]
            file_data = json.load(file)
            for entry in file_data:
                entry['story_id_and_dataset'] = f"{prefix}_{entry['story_id']}"
                entry['dataset'] = prefix
            data.extend(file_data)
    df = pd.json_normalize(data)
    df['author'] = df['story_id_and_dataset'].apply(extract_author)
    df['author'] = df['author'].str.lower()
    return df

def normalize_data(df, feature_cols, normalization='standard'):
    features = df[feature_cols].fillna(0)
    if normalization == 'standard':
        scaler = StandardScaler()
    elif normalization == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler:
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = features.values

    return features_scaled

def remove_outliers(df, features_scaled, contamination=0.05):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(features_scaled)
    df['outlier'] = outliers
    df_inliers = df[df['outlier'] == 1]
    features_inliers = features_scaled[df['outlier'] == 1]
    return df_inliers, features_inliers

def load_text_metrics_data(filepaths):
    """
    Loads data from JSON files and returns a DataFrame.
    """
    df = load_json_to_df(filepaths)
    return df

def calculate_features(df, feature_cols, normalization='standard', contamination=0.05, remove_outliers_flag=False):
    """
    Calculates DataFrame features and optionally removes outliers.
    """
    df = df.copy()  
    features_scaled = normalize_data(df, feature_cols, normalization)
    if remove_outliers_flag:
        df_inliers, features_inliers = remove_outliers(df, features_scaled, contamination)
    else:
        df_inliers = df
        features_inliers = features_scaled
    return df_inliers, features_inliers

def load_user_preferences(folder_path, feature_cols):
    """
    Loads all user preference CSVs into the folder 'folder_path'.
    Assume each CSV has columns in feature_cols.
    Returns a DataFrame with all users concatenated.
    """
    import glob
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []
    for file in all_files:
        tmp = pd.read_csv(file)
        df_list.append(tmp)
    user_df = pd.concat(df_list, ignore_index=True)
    
    user_df[feature_cols] = user_df[feature_cols].fillna(0)
    return user_df

def load_rankings(filepaths=config.rankings_filepath):
    ranking_dfs = []
    for rf in filepaths:
        df = pd.read_csv(rf)
        dataset = os.path.basename(rf).split('_')[0]
        df['dataset'] = dataset
        ranking_dfs.append(df)
    df_ranking = pd.concat(ranking_dfs, ignore_index=True)
    return df_ranking

def merge_ranking_and_metrics(metrics_filepath=config.stories_metrics_filepath, ranking_filepaths=config.rankings_filepath, feature_cols=config.selected_features):
    df_features = load_text_metrics_data(metrics_filepath)
    df_inliers, features_inliers = calculate_features(df_features, feature_cols=feature_cols)
    df_ranking = load_rankings(ranking_filepaths)
    df_merged = pd.merge(df_ranking, df_inliers, on='story_id', how='inner')
    return df_merged