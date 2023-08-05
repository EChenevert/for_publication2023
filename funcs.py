import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.utils import shuffle
import random
import statsmodels.api as sm
from sklearn import linear_model
import textwrap


def max_interquartile_outlierrm(df, target):
    """This function removes outliers from a DataFrame based on the maximum interquartile range (IQR).

    Outliers are identified as data points that fall above the upper boundary defined as Q3 + 1.5 * IQR, where
    Q3 is the third quartile and IQR is the interquartile range (the difference between Q3 and Q1).

    @params:
        df (DataFrame): The DataFrame from which outliers will be removed.
        target (str): The name of the target column for which outliers will be identified and removed.
        In our study it is accretion rate.
    @returns:
        DataFrame: A new DataFrame with outliers removed based on the specified target column.
    """
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    filtered_df = df[~(df[target] > (Q3 + 1.5 * IQR))]
    return filtered_df


def wrap_labels(ax, width, break_long_words=False):
    """
    From: https://medium.com/dunder-data/automatically-wrap-graph-labels-in-matplotlib-and-seaborn-a48740bc9ce

    Wrap tick labels on the x-axis of a matplotlib Axes object to fit within a specified width.
    The function takes an Axes object 'ax', the 'width' to wrap the labels, and an optional parameter
    'break_long_words' to control whether long words should be broken at the width or kept intact.

    @params:
        ax (matplotlib Axes): The Axes object for which the tick labels will be wrapped.
        width (int): The maximum width for the wrapped tick labels.
        break_long_words (bool, optional): If True, long words will be broken at the width. Default is False.

    Note:
        This function modifies the x-axis tick labels of the provided Axes 'ax' in place.

    Example usage:
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data)
        wrap_labels(ax, width=10, break_long_words=False)
        plt.show()
    """
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)


# https://www.analyticsvidhya.com/blog/2020/10/a-comprehensive-guide-to-feature-selection-using-wrapper-methods-in-python/#:~:text=1.-,Forward%20selection,with%20all%20other%20remaining%20features.
def backward_elimination(data, target, num_feats=5, significance_level=0.05):
    """
    Perform backward elimination for feature selection in a linear regression model.

    The function takes a DataFrame 'data' containing the independent features, a Series 'target'
    representing the dependent variable, an optional parameter 'num_feats' to specify the minimum
    number of features to retain, and an optional 'significance_level' for the p-values.

    @params:
        data (DataFrame): The DataFrame containing the independent features.
        target (Series): The Series representing the dependent variable.
        num_feats (int, optional): The minimum number of features to retain. Default is 5.
        significance_level (float, optional): The significance level for p-values. Default is 0.05.

    @returns:
        list: A list containing the selected features after backward elimination.

    Note:
        This function assumes that the 'data' DataFrame and 'target' Series are properly preprocessed
        and do not contain any missing values.
    """
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level) or (len(features) > num_feats):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features


def unscaled_weights_from_Xstandardized(X, bayesianReg: linear_model):
    """
    Informed from:
    https://stackoverflow.com/questions/57513372/can-i-inverse-transform-the-intercept-and-coefficients-of-
    https://stats.stackexchange.com/questions/74622/converting-standardized-betas-back-to-original-variables

    Get unscaled regression coefficients and the intercept from a Bayesian linear regression model
    when the input data 'X' has undergone standardization but the target variable remains in its original scale.

    The function takes a DataFrame 'X' containing the standardized independent features, and a Bayesian
    linear regression model 'bayesianReg' trained on the original scale target variable.

    @params:
        X (DataFrame): The DataFrame containing the standardized independent features.
        bayesianReg (linear_model): A trained Bayesian linear regression model.
    @returns:
        tuple: A tuple containing the unscaled regression coefficients and the intercept.

    Note:
        This function assumes that the 'X' DataFrame has been standardized, i.e., each column in 'X' has
        a mean of 0 and a standard deviation of 1. The 'bayesianReg' model should be trained on the original
        scale target variable, not on standardized targets.
    """
    a = bayesianReg.coef_
    i = bayesianReg.intercept_
    # Me tryna do my own thing
    coefs_new = []
    for x in range(len(X.columns)):
        col = X.columns.values[x]
        coefs_new.append((a[x] / (np.asarray(X.std()[col]))))
    intercept = i - np.sum(np.multiply(np.asarray(coefs_new), np.asarray(X.mean()/X.std())))  # hadamard product

    return coefs_new, intercept

def cv_results_and_plot(bay_model, bestfeatures, unscaled_predictor_matrix, predictor_matrix, target,
                        color_scheme: dict, marsh_key):
    """
    Perform cross-validation on a Bayesian linear regression model and provide various performance metrics
    and visualizations related to the model's predictions and residuals.

    @params:
        bay_model (BayesianModel): The Bayesian linear regression model.
        bestfeatures (list): The list of features selected by the feature selection process.
        unscaled_predictor_matrix (DataFrame): DataFrame containing the unscaled predictor matrix.
        predictor_matrix (DataFrame): DataFrame containing the scaled predictor matrix.
        target (DataFrame): DataFrame containing the target variable.
        color_scheme (dict): A dictionary containing color scheme for plots.
        marsh_key (str): A string representing the marsh key.
    @returns:
         dict: A dictionary with holding the scaled and unscaled weight coefficients, unscaled intercepts, unscaled
         regularization parameters, the number of well-determined weights, the standard deviation of teh predictions,
         the predictions, the residuals, and predictions for the residuals.
    """

    # Error Containers
    predicted = []  # holds they predicted values of y
    y_ls = []  # holds the true values of y
    residuals = []

    # Performance Metric Containers: I allow use the median because I want to be more robust to outliers
    r2_total_means = []  # holds the k-fold median r^2 value. Will be length of 100 due to 100 repeats
    mae_total_means = []  # holds the k-fold median Mean Absolute Error (MAE) value. Will be length of 100 due to 100 repeats

    # parameter holders
    weight_vector_ls = []  # holds the learned parameters for each k-fold test
    regularizor_ls = []  # holds the learned L2 regularization term for each k-fold test
    unscaled_w_ls = []  # holds the inverted weights to their natural scales
    intercept_ls = []  # holds the inverted intercept to its natural scale
    weight_certainty_ls = []  # holds the number of well-determinned parameters for each k-fold test
    prediction_certainty_ls = []  # holds the standard deviations of the predictions (predictive distributions)
    prediction_list = []

    for i in range(100):  # for 100 repeats
        try_cv = KFold(n_splits=5, shuffle=True)

        # Scaled lists
        r2_ls = []
        mae_ls = []

        # Certainty lists
        pred_certain = []
        pred_list = []
        w_certain = []

        for train_index, test_index in try_cv.split(predictor_matrix):
            X_train, X_test = predictor_matrix.iloc[train_index], predictor_matrix.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]
            # Fit the model
            bay_model.fit(X_train, y_train.values.ravel())
            # collect unscaled parameters
            unscaled_weights, intercept = unscaled_weights_from_Xstandardized(unscaled_predictor_matrix[bestfeatures],
                                                                              bay_model)
            # save
            unscaled_w_ls.append(unscaled_weights)

            intercept_ls.append(intercept)
            # Collect scaled parameters
            weights = bay_model.coef_
            weight_vector_ls.append(abs(weights))  # Take the absolute values of weights for relative feature importance
            regularizor = bay_model.lambda_ / bay_model.alpha_
            regularizor_ls.append(regularizor)
            design_m = np.asarray(X_train)
            eigs = np.linalg.eigh(bay_model.lambda_ * (design_m.T @ design_m))
            weight_certainty = []
            for eig in eigs[0]:
                weight_certainty.append(eig / (eig + bay_model.lambda_))
            weight_certainty = np.sum(weight_certainty)
            w_certain.append(weight_certainty)
            # Make our predictions for y
            ypred, ystd = bay_model.predict(X_test, return_std=True)

            pred_list += list(ypred)
            pred_certain += list(ystd)

            r2 = r2_score(y_test, ypred)
            r2_ls.append(r2)
            mae = mean_absolute_error(y_test, ypred)
            mae_ls.append(mae)

        # Average certainty in predictions
        prediction_certainty_ls.append(np.mean(pred_certain))
        prediction_list.append(pred_list)

        weight_certainty_ls.append(np.mean(w_certain))
        # Average predictions over the Kfold first: scaled
        r2_mean = np.mean(r2_ls)
        r2_total_means.append(r2_mean)
        mae_mean = np.mean(mae_ls)
        mae_total_means.append(mae_mean)

        predicted = predicted + list(cross_val_predict(bay_model, predictor_matrix, target.values.ravel(), cv=try_cv))
        residuals = residuals + list(target.values.ravel() - cross_val_predict(bay_model, predictor_matrix,
                                                                               target.values.ravel(), cv=try_cv))
        y_ls += list(target.values.ravel())

    # Add each of the model parameters to a dictionary
    weight_df = pd.DataFrame(weight_vector_ls, columns=bestfeatures)
    unscaled_weight_df = pd.DataFrame(unscaled_w_ls, columns=bestfeatures)

    # Now calculate the mean of th kfold means for each repeat: scaled accretion
    r2_final_median = np.mean(r2_total_means)
    mae_final_median = np.mean(mae_total_means)

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(9, 8))
    hb = ax.hexbin(x=y_ls,
                   y=predicted,
                   gridsize=30, edgecolors='grey',
                   cmap=color_scheme['cmap'], mincnt=1)
    ax.set_facecolor('white')
    ax.set_xlabel("Measured Accretion Rate (mm/yr)", fontsize=21)
    ax.set_ylabel("Estimated Accretion Rate (mm/yr)", fontsize=21)
    ax.set_title(marsh_key + " CRMS Stations", fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=18)
    cb = fig.colorbar(hb, ax=ax)
    cb.ax.get_yaxis().labelpad = 20
    cb.set_label('Density of Predictions', rotation=270, fontsize=21)

    ax.plot([target.min(), target.max()], [target.min(), target.max()],
            color_scheme['line'], lw=3)

    ax.annotate("Median r-squared = {:.3f}".format(r2_final_median), xy=(190, 30), xycoords='axes points',
                bbox=dict(boxstyle='round', fc='w'),
                size=15, ha='left', va='top')
    ax.annotate("Median MAE = {:.3f}".format(mae_final_median), xy=(190, 60), xycoords='axes points',
                bbox=dict(boxstyle='round', fc='w'),
                size=15, ha='left', va='top')
    # fig.savefig("__ENTER PATH TO SAVE" + marsh_key +
    #             "__ENTER FILENAME__.eps", format='eps',
    #             dpi=300,
    #             bbox_inches='tight')
    plt.show()

    # save all results in a dictionary
    dictionary = {
        "Scaled Weights": weight_df, "Unscaled Weights": unscaled_weight_df, "Unscaled Intercepts": intercept_ls,
        "Scaled regularizors": regularizor_ls, "# Well Determined Weights": weight_certainty_ls,
        "Standard Deviations of Predictions": prediction_certainty_ls, "Predictions": prediction_list,
        "Residuals": residuals, "Predicted for Residuals": predicted
    }

    # lets just look at the residuals.... why not right
    fig, ax = plt.subplots(figsize=(9, 7))
    hb = ax.hexbin(x=dictionary['Predicted for Residuals'],
                   y=dictionary['Residuals'],
                   gridsize=30, edgecolors='grey',
                   cmap='YlGnBu', mincnt=1)
    ax.set_facecolor('white')
    ax.set_xlabel("Fitted Value (Prediction)")
    ax.set_ylabel("Residual (y_true - y_predicted)")
    ax.set_title(marsh_key)
    cb = fig.colorbar(hb, ax=ax)
    cb.ax.get_yaxis().labelpad = 15
    cb.set_label('Density of Residuals', rotation=270)
    ax.axhline(0.0, linestyle='--')
    plt.show()

    return dictionary
