import pandas as pd
import os
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import itertools

def get_MAPE(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model using MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return mape


if __name__ == "__main__":
    iterative = False

    folder_path = 'optimization_data'
    all_files = sorted(os.listdir(folder_path)) # Get all files in the directory and sort them
    selected_files = [0]
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
            test_size = 0.3
            MAPEs_RF = []
            MAPEs_XGBoost = []
            feature_importances_all = {}
            X_train, X_test, y_train, y_test = train_test_split(features_data_onehot, runtime_data, test_size=test_size, random_state=0)
            print('Training data size: {}'.format(X_train.shape[0]))
            print('Testing data size: {}'.format(X_test.shape[0]))
            for run in range(1, num_repeats+1):
                if iterative: print('Run: {}'.format(run))
                # Generate the training and testing data
                X_train, X_test, y_train, y_test = train_test_split(features_data_onehot, runtime_data, test_size=test_size, random_state=run)

                # Testing the RF model
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                MAPE =get_MAPE(rf_model, X_test, y_test)
                if iterative: print('RF MAPE: {:.2f}'.format(MAPE))
                MAPEs_RF.append(MAPE)


                # # Generate all valid combinations of the one-hot encoded features
                # # First, create the one-hot encoded vectors for each category
                # categories = onehot_encoder.categories_ # Get the number of categories for each feature
                # onehot_vectors = []
                # for category in categories:
                #     onehot_vectors.append(np.eye(len(category)))
                # # Use Cartesian product to generate all possible combinations
                # combinations = list(itertools.product(*onehot_vectors))
                # # Flatten the combinations to get a single binary vector for each combination
                # flattened_combinations = ([list(itertools.chain(*combination)) for combination in combinations])
                # print("Total number of combinations: {}".format(len(flattened_combinations)))
                # # Predict the runtime for each configuration
                # predicting_combinations = [c for c in flattened_combinations if c not in X_train.tolist()]
                # print("Predicting {} combinations".format(len(predicting_combinations)))
                # predicted_runtimes = rf_model.predict(predicting_combinations)
                # minimum_runtime = min(predicted_runtimes)
                # minimum_runtime_index = np.argmin(predicted_runtimes)
                # print('Best optimization: {}, runtime: {}'.format(predicting_combinations[minimum_runtime_index], minimum_runtime))


                # Analyse the importance for each original code snippet and each LLM recommendation
                feature_importances = rf_model.feature_importances_
                # Compute SHAP values
                explainer = shap.TreeExplainer(rf_model)
                shap_values = explainer.shap_values(X_train)
                mean_shap_values = np.mean(shap_values, axis=0)

                # Store the feature names and importances into a dict for better visualization
                for i, feature in enumerate(feature_names):
                    code_snippet = "{}_{}".format(feature.split('_')[0], feature.split('_')[1])
                    LLM = feature.split('_')[2]
                    # print(code_snippet, LLM)
                    if code_snippet not in feature_importances_all:
                        feature_importances_all[code_snippet] = {}
                    feature_importances_all[code_snippet][LLM] = mean_shap_values[i]
                    # print('{} feature_importances: {}'.format(feature, feature_importances[i]))
                # print(feature_importances_all)

                # Print the selected LLMs for each code snippets
                for snippet in feature_importances_all:
                    print("Code Snippet: {}".format(snippet))
                    print(feature_importances_all[snippet])
                    shap_values_dict = feature_importances_all[snippet]
                    lowest_shap_feature = min(shap_values_dict, key=shap_values_dict.get)
                    lowest_shap_value = shap_values_dict[lowest_shap_feature]
                    print("{} leads to the lowest runtime: {}".format(lowest_shap_feature, lowest_shap_value))



                # # SHAP analysis
                # explainer = shap.TreeExplainer(rf_model)
                # shap_values = explainer.shap_values(X_train)
                # # Plot SHAP values for the most important feature
                # shap.summary_plot(shap_values, X_train, feature_names=feature_names)


                # # Testing the XGBoost model
                # xgb_model = XGBRegressor(n_estimators=100, random_state=42)
                # xgb_model.fit(X_train, y_train)
                # MAPE =get_MAPE(xgb_model, X_test, y_test)
                # if iterative: print('XGBoost MAPE: {:.2f}'.format(MAPE))
                # MAPEs_XGBoost.append(MAPE)

            print('\nAverage RF MAPE: {:.2f}'.format(np.mean(MAPEs_RF)))
            # print('Average XGBoost MAPE: {:.2f}\n'.format(np.mean(MAPEs_XGBoost)))