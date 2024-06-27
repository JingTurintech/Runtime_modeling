import pandas as pd
import os
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
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
    test_size = 0
    seletected_performance_metrics = ['runtime', 'cpu', 'memory']
    current_performance = 'runtime'
    seletected_accuracy_metrics = ['RMSE', 'MAPE']
    selected_models = ['RF', 'BayesianRR']
    folder_path = 'optimization_data'
    all_files = sorted(os.listdir(folder_path)) # Get all files in the directory and sort them
    selected_files = range(len(all_files)) # Selected all or a subset of files
    # selected_files = [1]  # Selected all or a subset of files

    # Create a dictionary to store the results
    results_to_save = {
        'Dataset': [],
        'Model': [],
    }
    for performance in seletected_performance_metrics:
        for accuracy in seletected_accuracy_metrics:
            results_to_save['{} {}'.format(performance, accuracy)] = []
            results_to_save['{} {} stats'.format(performance, accuracy)] = []

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

            # # One-hot encode the features
            # dropped_categories = ["original"] * features_data.shape[1]
            # onehot_encoder = OneHotEncoder(sparse_output=False, drop=dropped_categories)
            # features_data_onehot = onehot_encoder.fit_transform(features_data)
            # feature_names = onehot_encoder.get_feature_names_out(input_features=selected_feature_names)
            # print('dimensions after onehot: {}'.format(features_data_onehot.shape[1]))

            # Label encode the features
            label_encoder = LabelEncoder()
            features_data_onehot = pd.DataFrame(features_data)
            for col in features_data_onehot.columns:
                features_data_onehot[col] = LabelEncoder().fit_transform(features_data_onehot[col])
            print('dimensions after label encoding: {}'.format(features_data_onehot.shape[1]))

            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_poly = poly.fit_transform(features_data_onehot)
            print('dimensions after polynominal features: {}'.format(X_poly.shape[1]))
            # print(X_poly.shape)

            # Get the new feature names
            new_feature_names = selected_feature_names.copy()  # Start with original feature names
            for i in range(len(selected_feature_names)):
                for j in range(i + 1, len(selected_feature_names)):
                    new_feature_names.append(f'{selected_feature_names[i]} * {selected_feature_names[j]}')  # Interaction terms

            # # Create a DataFrame with the new feature names
            # new_features_df = pd.DataFrame(X_poly, columns=new_feature_names)
            # print(new_features_df)

            performance_data = data_frame[current_performance].to_numpy()

            # Create a Bayesian Ridge regression model
            model = BayesianRidge()

            # Fit the model
            model.fit(X_poly, performance_data)

            # Get the coefficients (influence) of each feature
            feature_influence = model.coef_

            max_coe = np.max(np.abs(feature_influence))
            fea_name = new_feature_names[np.argmax(feature_influence)]

            # Print the influence of each feature
            for i, feature_name in enumerate(new_feature_names):
                print(f"{feature_name}: {feature_influence[i]:.4f}")

            print('{} max feature coefficient: {}'.format(fea_name, max_coe))


if __name__ == "__main__":
    main()