import pandas as pd
import os
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import itertools
from sklearn.preprocessing import MinMaxScaler
from utils import get_RMSE, get_MAPE, interpretate_cohen_d

def main():
    iterative = False
    save_results = True
    total_runs = 30
    test_size = 0.2
    seletected_performance_metrics = ['runtime', 'cpu', 'memory']
    seletected_accuracy_metrics = ['RMSE', 'MAPE']
    selected_models = ['RF', 'BayesianRR']
    folder_path = 'optimization_data'
    all_files = sorted(os.listdir(folder_path)) # Get all files in the directory and sort them
    # selected_files = range(len(all_files)) # Selected all or a subset of files
    selected_files = [11]  # Selected all or a subset of files

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
            results_to_save['Dataset'].append(file_name.replace('.csv', ''))

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

            # # Label encode the features
            # label_encoder = LabelEncoder()
            # features_data_onehot = pd.DataFrame(features_data)
            # for col in features_data_onehot.columns:
            #     features_data_onehot[col] = LabelEncoder().fit_transform(features_data_onehot[col])
            # print('dimensions after label encoding: {}'.format(features_data_onehot.shape[1]))


            for current_performance in seletected_performance_metrics:
                print('\n---Performance metric: {}---'.format(current_performance))
                # Get the performance labels
                performance_data = data_frame[current_performance].to_numpy()
                # print('performance_data: {}'.format(performance_data))

                # Split the whole dataset into train and test set
                X_train, X_test, y_train, y_test = train_test_split(features_data_onehot, performance_data, test_size=test_size, random_state=0)
                print('Training data size: {}'.format(X_train.shape[0]))
                print('Testing data size: {}'.format(X_test.shape[0]))

                # Repeated evaluation to avoid randomness
                results_all_runs = {} # Dict to store the results for all runs
                for current_model in selected_models: # Initialize the dict
                    for current_accuracy in seletected_accuracy_metrics:
                        results_all_runs['{} {}'.format(current_model, current_accuracy)] = []
                print('total_runs: {}'.format(total_runs))
                for run in range(1, total_runs + 1):
                    if iterative: print('Run: {}'.format(run))
                    # Generate the training and testing data
                    X_train, X_test, y_train, y_test = train_test_split(features_data_onehot, performance_data, test_size=test_size, random_state=run)

                    # # Create a MinMaxScaler object and normalize the performance
                    # scaler = MinMaxScaler()
                    # scaler.fit_transform(y_train)
                    # scaler.fit(y_test)

                    for current_model in selected_models:
                        if current_model == 'RF':
                            model = RandomForestRegressor(n_estimators=100)

                        elif current_model == 'BayesianRR':
                            model = BayesianRidge()

                        # Train mode
                        model.fit(X_train, y_train)

                        for current_accuracy in seletected_accuracy_metrics:
                            if current_accuracy == 'RMSE':
                                RMSE = get_RMSE(model, X_test, y_test)
                                if iterative: print('{} {}: {:.2f}'.format(current_model, current_accuracy, RMSE))
                                results_all_runs['{} {}'.format(current_model, current_accuracy)].append(RMSE)

                            elif current_accuracy == 'MAPE':
                                MAPE = get_MAPE(model, X_test, y_test)
                                if iterative: print('{} {}: {:.2f}'.format(current_model, current_accuracy, MAPE))
                                results_all_runs['{} {}'.format(current_model, current_accuracy)].append(MAPE)

                # After all runs finished, perform the Wilcoxon Signed-Rank Test, t-test and Cohen's d to validate the significance of the difference
                for current_accuracy in seletected_accuracy_metrics:
                    for model1, model2 in itertools.combinations(selected_models, 2):
                        results_model1 = np.array(results_all_runs['{} {}'.format(model1, current_accuracy)])
                        results_model2 = np.array(results_all_runs['{} {}'.format(model2, current_accuracy)])

                        # Perform the Wilcoxon Signed-Rank Test
                        w_statistic, w_p_value = stats.wilcoxon(results_model1, results_model2, alternative='two-sided')

                        # Perform paired t-test
                        t_statistic, t_p_value = ttest_rel(results_model1, results_model2)

                        # Calculate Cohen's d effect size
                        mean_difference = np.mean(results_model1 - results_model2) # Calculate mean difference in error rates
                        diff = results_model1 - results_model2 # Calculate pooled standard deviation
                        pooled_std = np.std(diff, ddof=1)
                        cohen_d = mean_difference / pooled_std

                        print('\n> Average {} {} {}: {:.2f}'.format(model1, current_performance, current_accuracy, np.mean(results_all_runs['{} {}'.format(model1, current_accuracy)])))
                        print('> Average {} {} {}: {:.2f}'.format(model2, current_performance, current_accuracy, np.mean(results_all_runs['{} {}'.format(model2, current_accuracy)])))
                        print('> {} Wilcoxon signed Rank test p value: {:.2f}'.format(current_accuracy, w_p_value))
                        print('> {} t-test p value: {:.2f}'.format(current_accuracy, t_p_value))
                        print('> Cohen_d value: {:.2f}'.format(cohen_d))
                        interpretate_cohen_d(cohen_d, model1, model2)

                    results_to_save['{} {} stats'.format(current_performance, current_accuracy)] += ['w_p:{}, t_p:{}'.format(round(w_p_value, 2), round(t_p_value, 2)), 'cohen:{}'.format(round(cohen_d, 2))]

                for current_accuracy in seletected_accuracy_metrics:
                    for current_model in selected_models:
                        results_to_save['{} {}'.format(current_performance, current_accuracy)].append(round(np.mean(results_all_runs['{} {}'.format(current_model, current_accuracy)]), 2))
                        if len(results_to_save['Model']) < len(results_to_save['{} {}'.format(current_performance, current_accuracy)]):
                            results_to_save['Model'].append(current_model)
                        if len(results_to_save['Dataset']) < len(results_to_save['{} {}'.format(current_performance, current_accuracy)]):
                            results_to_save['Dataset'].append('')


    for temp in results_to_save:
        print(temp, len(results_to_save[temp]))
    # Create a DataFrame from the dictionary
    results_df = pd.DataFrame(results_to_save)
    # Print the results
    print("\nResults:")
    print(results_df)
    if save_results:
        # Save the results to a CSV file (adjust the file path as needed)
        results_df.to_csv("results/{}_vs_{}_results.csv".format(selected_models[0], selected_models[1]), index=False)



if __name__ == "__main__":
    main()