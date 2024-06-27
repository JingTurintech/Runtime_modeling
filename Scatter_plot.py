import pandas as pd
import os
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.manifold import TSNE
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import itertools

def main():
    iterative = False
    save_results = True
    total_runs = 30
    test_size = 0.2
    seletected_performance_metrics = ['runtime'] # 'runtime', 'cpu', 'memory'
    folder_path = 'optimization_data'
    all_files = sorted(os.listdir(folder_path)) # Get all files in the directory and sort them
    selected_files = range(len(all_files)) # Selected all or a subset of files
    # selected_files = [1]  # Selected all or a subset of files


    for i_file, file_name in enumerate([all_files[i] for i in selected_files]):
        if file_name.endswith('.csv'):
            # Reading dataset from csv files
            file_path = os.path.join(folder_path, file_name)
            data_frame = pd.read_csv(file_path)  # Read the CSV file
            data_frame = data_frame[(data_frame[seletected_performance_metrics] != 0).all(axis=1)] # Removes samples with zero values for performance, which may be caused by bugs
            print('\n>> Dataset {}: {}'.format(selected_files[i_file], file_name))
            data_array = data_frame.values # Remove the header
            n_samples = data_array.shape[0]
            print('n_samples: {}'.format(n_samples))
            print('total_n_features: {}'.format(data_array.shape[1]))

            # Only select the features starts with 'structure'
            header_list = list(data_frame.columns)
            selected_features = [i for i in range(len(header_list)) if header_list[i].startswith('structure')]
            selected_feature_names = [header_list[i] for i in selected_features]
            print('selected_features: {}'.format(selected_feature_names))
            print('dimensions of selected_features: {}'.format(len(selected_features)))
            features_data = data_array[:, selected_features]

            # One-hot encode the features
            dropped_categories = ["original"] * features_data.shape[1]
            onehot_encoder = OneHotEncoder(sparse_output=False, drop=dropped_categories)
            features_data_onehot = onehot_encoder.fit_transform(features_data)
            feature_names = onehot_encoder.get_feature_names_out(input_features=selected_feature_names)
            print('dimensions after onehot: {}'.format(features_data_onehot.shape[1]))

            # # Initialize t-SNE with 1 component (1-dimensional projection)
            # tsne = TSNE(n_components=1, random_state=42)
            # # Project features into 1-dimensional space
            # projected_data = tsne.fit_transform(features_data_onehot)
            #
            # for current_performance in seletected_performance_metrics:
            #     print('\n---Performance metric: {}---'.format(current_performance))
            #     # Get the performance labels
            #     runtime_data = data_frame[current_performance].to_numpy()
            #     # print('runtime_data: {}'.format(runtime_data))
            #
            #     plt.figure(figsize=(10, 6))
            #     plt.scatter(projected_data, data_frame[current_performance], alpha=0.5)
            #     plt.xlabel("Projected Feature (t-SNE)")
            #     plt.ylabel(current_performance)
            #     plt.title("Dataset: {}".format(file_name))
            #     plt.show()

            tsne_2d = TSNE(n_components=2, random_state=42)
            projected_data_2d = tsne_2d.fit_transform(features_data_onehot)

            for current_performance in seletected_performance_metrics:
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, projection='3d')

                # Scatter plot in 3D
                ax.scatter(projected_data_2d[:, 0], projected_data_2d[:, 1], data_frame[current_performance], alpha=0.5)

                # Set the view point (elev and azim)
                ax.view_init(elev=15, azim=45)  # Adjust these angles as needed

                # Customize labels
                ax.set_xlabel("t-NSE Feature 1")
                ax.set_ylabel("t-SNE Feature 2")
                ax.set_zlabel(current_performance)

                # Show the plot
                plt.title("Dataset: {}".format(file_name))
                plt.show()

if __name__ == "__main__":
    main()