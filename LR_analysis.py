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
import itertools

def get_MAPE(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model using MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return mape

def main():
    iterative = False

    folder_path = 'optimization_data'
    all_files = sorted(os.listdir(folder_path)) # Get all files in the directory and sort them
    selected_files = range(len(all_files))
    for file_name in [all_files[i] for i in selected_files]:
        if file_name.endswith('.csv'):
            # Reading dataset from csv files
            file_path = os.path.join(folder_path, file_name)
            data_frame = pd.read_csv(file_path)  # Read the CSV file
            print('\nDataset: {}'.format(file_name))
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
            onehot_encoder = OneHotEncoder(sparse_output=False)
            features_data_onehot = onehot_encoder.fit_transform(features_data)
            feature_names = onehot_encoder.get_feature_names_out(input_features=selected_feature_names)
            print('dimensions after onehot: {}'.format(features_data_onehot.shape[1]))

            # Get the performance labels
            runtime_data = data_frame['runtime'].to_numpy()
            # print('runtime_data: {}'.format(runtime_data))

            # Repeated evaluation to avoid randomness
            num_repeats = 1
            test_size = 0.001
            MAPEs_RF = []
            MAPEs_XGBoost = []
            feature_importances_all = {}
            X_train, X_test, y_train, y_test = train_test_split(features_data_onehot, runtime_data, test_size=test_size,
                                                                random_state=0)
            print('Training data size: {}'.format(X_train.shape[0]))
            print('Testing data size: {}'.format(X_test.shape[0]))
            for run in range(1, num_repeats + 1):
                if iterative: print('Run: {}'.format(run))
                # Generate the training and testing data
                X_train, X_test, y_train, y_test = train_test_split(features_data_onehot, runtime_data,
                                                                    test_size=test_size, random_state=run)

                # Fit a linear regression model
                model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

                # Get the results
                coefficients = model.params
                standard_errors = model.bse
                for i, feature_name in enumerate(feature_names):
                    print(
                        f"Feature: {feature_name}, Coefficient: {coefficients[i]}, Standard Error: {standard_errors[i]}")



if __name__ == "__main__":
    main()