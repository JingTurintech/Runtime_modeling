import pandas as pd
import os
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools

def main():
    total_runs = 30
    test_size = 0.2
    seletected_performance_metrics = ['runtime', 'cpu', 'memory']
    seletected_accuracy_metrics = ['RMSE', 'MAPE']
    selected_models = ['RF', 'BayesianRR']
    folder_path = 'optimization_data'
    all_files = sorted(os.listdir(folder_path)) # Get all files in the directory and sort them
    selected_files = range(len(all_files))  # Selected all or a subset of files
    # selected_files = [0, 2]  # Selected all or a subset of files

    # Create a DataFrame to store the information
    info_df = {
        'Dataset': [],
        'N_total_samples': [],
        'N_train_samples': [],
        'N_test_samples': [],
        'N_selected_features': [],
        'N_onehot_features': []
    }

    for i_file, file_name in enumerate([all_files[i] for i in selected_files]):
        if file_name.endswith('.csv'):
            # Reading dataset from csv files
            file_path = os.path.join(folder_path, file_name)
            data_frame = pd.read_csv(file_path)  # Read the CSV file
            data_frame = data_frame[(data_frame[seletected_performance_metrics] != 0).all(
                axis=1)]  # Removes samples with zero values for performance, which may be caused by bugs
            print('\n>> Dataset {}: {}'.format(selected_files[i_file], file_name))
            data_array = data_frame.values  # Remove the header
            n_samples = data_array.shape[0]
            total_n_features = data_array.shape[1]
            print('n_samples: {}'.format(n_samples))
            print('total_n_features: {}'.format(total_n_features))

            # Only select the features starts with 'structure'
            header_list = list(data_frame.columns)
            selected_features = [i for i in range(len(header_list)) if header_list[i].startswith('structure')]
            selected_feature_names = [header_list[i] for i in selected_features]
            n_selected_features = len(selected_features)
            print('selected_features: {}'.format(selected_feature_names))
            print('dimensions of selected_features: {}'.format(n_selected_features))
            features_data = data_array[:, selected_features]

            # One-hot encode the features
            dropped_categories = ["original"] * features_data.shape[1]
            onehot_encoder = OneHotEncoder(sparse_output=False, drop=dropped_categories)
            features_data_onehot = onehot_encoder.fit_transform(features_data)
            feature_names = onehot_encoder.get_feature_names_out(input_features=selected_feature_names)
            n_onehot_features = features_data_onehot.shape[1]
            print('dimensions after onehot: {}'.format(n_onehot_features))
            runtime_data = data_frame['runtime'].to_numpy()

            # Split the whole dataset into train and test set
            X_train, X_test, y_train, y_test = train_test_split(features_data_onehot, runtime_data, test_size=test_size,
                                                                random_state=0)

            n_train_samples = X_train.shape[0]
            n_test_samples = X_test.shape[0]

            # Append the information for the current dataset
            info_df['Dataset'].append('http://dev.artemis.turintech.ai/optimisations/{}'.format(file_name.replace('.csv', '')))
            info_df['N_total_samples'].append(n_samples)
            info_df['N_train_samples'].append(n_train_samples)
            info_df['N_test_samples'].append(n_test_samples)
            info_df['N_selected_features'].append(n_selected_features)
            info_df['N_onehot_features'].append(n_onehot_features)

            # Save the information to a CSV file
            pd.DataFrame(info_df).to_csv('results/dataset_info.csv', index=False)

            # Print a success message
            print(f"Dataset information saved to dataset_info.csv")

if __name__ == "__main__":
    main()
