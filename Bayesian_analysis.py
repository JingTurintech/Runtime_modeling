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

def get_MAPE(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model using MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return mape


def get_RMSE(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model using RMSE
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    return rmse

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
            dropped_categories = ["original"] * features_data.shape[1]
            onehot_encoder = OneHotEncoder(sparse_output=False, drop=dropped_categories)
            features_data_onehot = onehot_encoder.fit_transform(features_data)
            feature_names = onehot_encoder.get_feature_names_out(input_features=selected_feature_names)
            print('dimensions after onehot: {}'.format(features_data_onehot.shape[1]))

            # Get the performance labels
            runtime_data = data_frame['runtime'].to_numpy()
            # print('runtime_data: {}'.format(runtime_data))

            # Repeated evaluation to avoid randomness
            num_repeats = 1
            test_size = 0.2
            MAPEs_RF = []
            MAPEs_BayesianLR = []
            RMSEs_BayesianLR = []
            RMSEs_RF = []
            X_train, X_test, y_train, y_test = train_test_split(features_data_onehot, runtime_data, test_size=test_size,
                                                                random_state=0)
            print('Training data size: {}'.format(X_train.shape[0]))
            print('Testing data size: {}'.format(X_test.shape[0]))
            for run in range(1, num_repeats + 1):
                if iterative: print('Run: {}'.format(run))
                # Generate the training and testing data
                X_train, X_test, y_train, y_test = train_test_split(features_data_onehot, runtime_data,
                                                                    test_size=test_size, random_state=run)

                # Create a Bayesian Ridge regression model
                model = BayesianRidge()

                # Fit the model
                model.fit(X_train, y_train)

                """
                Get model.sigma_, which provides the standard deviation (uncertainty) of the noise term (residuals) in the data.

                Diagonal Elements (Variances):
                The diagonal elements represent the uncertainty (variance) associated with each individual feature’s weight (coefficient).
                Larger variances imply more uncertainty about the contribution of that feature to the predictions.
                Smaller variances indicate more confidence in the weight estimate.

                Off-Diagonal Elements (Covariances):
                The off-diagonal elements represent the covariance between pairs of features.
                Positive covariances indicate that when one feature’s weight increases, the other feature’s weight tends to increase as well.
                Negative covariances suggest an inverse relationship.
                Strong covariances may indicate collinearity (high correlation) between features.
                """
                feature_variances = model.sigma_
                # Extract the diagonal elements (variances) as specific uncertainties for each feature
                specific_uncertainties = np.diag(feature_variances)
                # Print the specific uncertainties
                for i, uncertainty in enumerate(specific_uncertainties):
                    print(f"Feature {i + 1} uncertainty: {uncertainty:.4f}")

                # Get model.lambda_, which quantifies the precision (inverse variance) of the estimated coefficients in Bayesian linear regression.
                # Higher values of lambda (>=100) indicate more confidence (less uncertainty) in the weights.
                coefficient_precision = model.lambda_
                print("Precision of coefficients: {}".format(coefficient_precision))

                # Get model.alpha_, which contains the estimated precision of the noise in the target variable.
                # A higher value of alpha_ (>=100) suggests that the model assumes the observed data points are close to the true values (low noise), while a lower value suggests more noise in the observations.
                noise_precision = model.alpha_
                print("Precision of noise: {}".format(noise_precision))

                # Examine the prediction and uncertainty of Bayesian LR
                y_pred_mean, y_pred_var = model.predict(X_test, return_std=True)
                # Print the uncertainty for each sample prediction
                for i, (mean, var) in enumerate(zip(y_pred_mean, y_pred_var)):
                    print(f"Sample {i + 1} prediction: {mean:.4f}, Uncertainty (variance): {var:.4f}")


            #     # Testing the BayesianLR model
            #     blr_model = BayesianRidge()
            #     MAPE = get_MAPE(blr_model, X_test, y_test)
            #     if iterative: print('BayesianLR MAPE: {:.2f}'.format(MAPE))
            #     MAPEs_BayesianLR.append(MAPE)
            #     RMSE = get_RMSE(blr_model, X_test, y_test)
            #     if iterative: print('BayesianLR RMSE: {:.2f}'.format(RMSE))
            #     RMSEs_BayesianLR.append(RMSE)
            #
            #     # Testing the RF model
            #     rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            #     MAPE =get_MAPE(rf_model, X_test, y_test)
            #     if iterative: print('RF MAPE: {:.2f}'.format(MAPE))
            #     MAPEs_RF.append(MAPE)
            #     RMSE =get_RMSE(rf_model, X_test, y_test)
            #     if iterative: print('RF RMSE: {:.2f}'.format(RMSE))
            #     RMSEs_RF.append(RMSE)
            #
            # # Perform the Wilcoxon Signed-Rank Test
            # W, p_value = stats.wilcoxon(MAPEs_BayesianLR, MAPEs_RF, alternative='two-sided')
            # print('\nAverage BayesianLR MAPE: {}'.format(np.mean(MAPEs_BayesianLR)))
            # print('Average RF MAPE: {}'.format(np.mean(MAPEs_RF)))
            # print('MAPE p-value: {}'.format(p_value))
            #
            # W, p_value = stats.wilcoxon(RMSEs_BayesianLR, RMSEs_RF, alternative='two-sided')
            # print('\nAverage BayesianLR RMSE: {}'.format(np.mean(RMSEs_BayesianLR)))
            # print('Average RF RMSE: {}'.format(np.mean(RMSEs_RF)))
            # print('RMSE p-value: {}'.format(p_value))



if __name__ == "__main__":
    main()