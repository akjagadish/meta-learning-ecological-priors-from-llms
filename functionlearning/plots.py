from wordcloud import WordCloud
from collections import Counter
import torch.nn.functional as F
from groupBMC.groupBMC import GroupBMC
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from os import getenv
from dotenv import load_dotenv
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import warnings
from scipy.optimize import OptimizeWarning
load_dotenv()
SYS_PATH = getenv('BERMI_DIR')
PARADIGM_PATH = f"{SYS_PATH}/functionlearning"
sys.path.append(PARADIGM_PATH)
sys.path.append(f"{PARADIGM_PATH}/data")
FONTSIZE = 20


def plot_functionlearning_data_statistics(mode=0):

    def calculate_bic(n, rss, num_params):
        bic = n * np.log(rss/n) + num_params * np.log(n)
        # if bic is an array return item else return bic
        return bic.item() if isinstance(bic, np.ndarray) else bic

    def return_data_stats(data, poly_degree=2, first=False):

        df = data.copy()
        max_tasks = 400
        max_trial = 20
        # sample tasks
        tasks = range(0, max_tasks) if first else np.random.choice(
            df.task_id.unique(), max_tasks, replace=False)
        all_corr, all_bics_linear, all_bics_quadratic, gini_coeff, all_accuraries_linear, all_accuraries_polynomial = [], [], [], [], [], []
        all_features_without_norm, all_features_with_norm, all_targets_with_norm = np.array(
            []), np.array([]), np.array([])
        for i in tasks:
            df_task = df[df['task_id'] == i]
            if len(df_task) > 0:  # arbitary data size threshold
                y = df_task['target'].to_numpy()
                X = df_task["input"].to_numpy()
                X = np.stack(X)
                X = (X - X.min())/(X.max() - X.min() + 1e-6)
                y = (y - y.min())/(y.max() - y.min() + 1e-6)

                all_features_with_norm = np.concatenate(
                    [all_features_with_norm, X.flatten()])
                all_targets_with_norm = np.concatenate(
                    [all_targets_with_norm, y.flatten()])

                if (y == 0).all() or (y == 1).all():
                    pass
                else:
                    # X_linear = PolynomialFeatures(1).fit_transform(X)
                    X_linear = PolynomialFeatures(
                        1, include_bias=False).fit_transform(X)

                    # # linear regression from X_linear to y
                    # linear_regresion = sm.OLS(y, X_linear).fit()

                    # # polinomial regression from X_poly
                    # X_poly = PolynomialFeatures(
                    #     poly_degree, interaction_only=True, include_bias=False).fit_transform(X)
                    # polynomial_regression = sm.OLS(y, X_poly).fit()

                    # fit gaussian process with linear kernel to X_linear and y
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    from sklearn.gaussian_process.kernels import RBF, DotProduct

                    GP_linear = GaussianProcessRegressor(
                        kernel=1.0 * DotProduct(), n_restarts_optimizer=10)
                    GP_linear.fit(X_linear, y)

                    # fit gaussian process with rbf kernel to X_poly and y
                    GP_quadratic = GaussianProcessRegressor(
                        kernel=1.0 * RBF(), n_restarts_optimizer=10)
                    GP_quadratic.fit(X_linear, y)

                    rss = np.sum((y - GP_linear.predict(X_linear))**2)
                    all_bics_linear.append(calculate_bic(
                        len(y), rss, 1))  # linear_regresion.bic)

                    rss = np.sum((y - GP_quadratic.predict(X_linear))**2)
                    all_bics_quadratic.append(calculate_bic(
                        X_linear.shape[0], rss, len(GP_quadratic.kernel.theta)))
                    # polynomial_regression.bic)

                    if X.shape[0] < max_trial:
                        pass
                    else:
                        task_accuraries_linear = []
                        # task_accuraries_polynomial = []
                        for trial in range(max_trial):
                            X_linear_uptotrial = X_linear[:trial]
                            # X_poly_uptotrial = X_poly[:trial]
                            y_uptotrial = y[:trial]

                            if (y_uptotrial == 0).all() or (y_uptotrial == 1).all() or trial == 0:
                                task_accuraries_linear.append(1.)
                                # task_accuraries_polynomial.append(0.5)
                            else:

                                # linear regression prediction
                                linear_reg = sm.OLS(
                                    y_uptotrial, X_linear_uptotrial).fit()
                                # log_reg_quadratic = sm.OLS(y_uptotrial, X_poly_uptotrial).fit(method='bfgs', maxiter=10000, disp=0)

                                y_linear_trial = linear_reg.predict(
                                    X_linear[trial])
                                # y_poly_trial = log_reg_quadratic.predict(X_poly[trial])

                                # mean squared error
                                task_accuraries_linear.append(
                                    float((y_linear_trial - y[trial]).item())**2)
                            # task_accuraries_polynomial.append(float((y_poly_trial.round() == y[trial]).item()))

                    all_accuraries_linear.append(task_accuraries_linear)
                    # all_accuraries_polynomial.append(task_accuraries_polynomial)
        all_accuraries_linear = np.array(all_accuraries_linear).mean(0)
        # all_accuraries_polynomial = np.array(all_accuraries_polynomial).mean(0)

        logprobs = torch.from_numpy(-0.5 *
                                    np.stack((all_bics_linear, all_bics_quadratic), -1))
        joint_logprob = logprobs + \
            torch.log(torch.ones([]) / logprobs.shape[1])
        marginal_logprob = torch.logsumexp(joint_logprob, dim=1, keepdim=True)
        posterior_logprob = joint_logprob - marginal_logprob

        return all_corr, gini_coeff, posterior_logprob, all_accuraries_linear, all_accuraries_polynomial, all_targets_with_norm, all_features_with_norm

    # set env_name and color_stats based on mode
    if mode == 0:
        env_name = f'{SYS_PATH}/functionlearning/data/generated_tasks/claude_generated_functionlearningtasks_paramsNA_dim1_data20_tasks9991_run0_procid0_pversion2'
        color_stats = '#405A63'  # '#2F4A5A'# '#173b4f'
    elif mode == 1:  # last plot
        env_name = f'{SYS_PATH}/functionlearning/data/generated_tasks/synthetic_functionlearning_tasks_dim1_data25_tasks10000'
        color_stats = '#66828F'  # 5d7684'# '#5d7684'
    elif mode == 2:  # first plot
        env_name = f'{SYS_PATH}/functionlearning/data/generated_tasks/real_data'
        color_stats = '#173b4f'  # '#0D2C3D' #'#8b9da7'
    # elif mode == 3:
    #     env_name = f'{SYS_PATH}/functionlearning/data/synthetic_tasks_dim4_data650_tasks1000_nonlinearTrue'
    #     color_stats = '#5d7684'

    # load data
    data = pd.read_csv(f'{env_name}.csv')
    data.input = data['input'].apply(lambda x: np.array(eval(x)))
    if mode == 2:
        data.target = data['target'].apply(lambda x: np.array(eval(x)))

    if os.path.exists(f'{SYS_PATH}/functionlearning/data/stats/stats_{str(mode)}.npz'):
        stats = np.load(
            f'{SYS_PATH}/functionlearning/data/stats/stats_{str(mode)}.npz', allow_pickle=True)
        all_corr, gini_coeff, posterior_logprob, all_accuraries_linear = stats['all_corr'], stats[
            'gini_coeff'], stats['posterior_logprob'], stats['all_accuraries_linear']
        all_accuraries_polynomial, all_targets_with_norm, all_features_with_norm = stats[
            'all_accuraries_polynomial'], stats['all_targets_with_norm'], stats['all_features_with_norm']
    else:
        all_corr, gini_coeff, posterior_logprob, all_accuraries_linear, all_accuraries_polynomial, \
            all_targets_with_norm, all_features_with_norm = return_data_stats(
                data)
    # gini_coeff = np.array(gini_coeff)
    # gini_coeff = gini_coeff[~np.isnan(gini_coeff)]
    posterior_logprob = posterior_logprob[:, 0].exp().detach().numpy()

    FONTSIZE = 22  # 8
    fig, axs = plt.subplots(1, 4,  figsize=(6*4, 4))  # figsize=(6.75, 1.5))
    axs[0].plot(all_accuraries_linear, color=color_stats, alpha=1., lw=3)
    # axs[0].plot(all_accuraries_polynomial, alpha=0.7)
    sns.histplot(np.array(all_features_with_norm), ax=axs[1], bins=11, binrange=(
        0., 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    sns.histplot(np.stack(all_targets_with_norm).reshape(-1), ax=axs[2], bins=11, binrange=(
        0., 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    sns.histplot(posterior_logprob, ax=axs[3], bins=5, binrange=(
        0.0, 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    # axs[1].set_xlim(-1, 1)

    axs[0].set_ylim(0., 1.05)
    axs[1].set_ylim(0,  0.2)
    axs[2].set_ylim(0,  0.2)
    axs[2].set_xlim(0., 1.05)
    axs[3].set_xlim(0., 1.05)

    # axs[0].set_yticks(np.arange(0.5, 1.05, 0.25))
    # axs[1].set_yticks(np.arange(0, 0.45, 0.2))
    # axs[2].set_yticks(np.arange(0, 0.4, 0.15))
    axs[3].set_yticks(np.arange(0, 1.05, 0.5))

    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[3].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)

    axs[0].set_ylabel('Squared Error', fontsize=FONTSIZE)
    axs[1].set_ylabel('Proportion', fontsize=FONTSIZE)
    axs[2].set_ylabel('')
    axs[3].set_ylabel('')

    if mode == 3:
        axs[0].set_xlabel('Trials', fontsize=FONTSIZE)
        axs[1].set_xlabel('Input', fontsize=FONTSIZE)
        axs[2].set_xlabel('Target', fontsize=FONTSIZE)
        axs[3].set_xlabel('Posterior probability ', fontsize=FONTSIZE)

    # set title
    if mode == 2:
        axs[0].set_title('Performance', fontsize=FONTSIZE)
        axs[1].set_title('Input distribution', fontsize=FONTSIZE)
        axs[2].set_title('Target distribution', fontsize=FONTSIZE)
        axs[3].set_title('Linearity', fontsize=FONTSIZE)

    plt.tight_layout()
    sns.despine()
    plt.savefig(f'{SYS_PATH}/figures/functionlearning_stats_' +
                str(mode) + '.svg', bbox_inches='tight')
    plt.show()

    # # save corr, gini, posterior_logprob, and all_accuraries_linear for each mode in one .npz file
    if not os.path.exists(f'{SYS_PATH}/functionlearning/data/stats/stats_{str(mode)}.npz'):
        np.savez(f'{SYS_PATH}/functionlearning/data/stats/stats_{str(mode)}.npz', all_corr=all_corr, gini_coeff=gini_coeff, posterior_logprob=posterior_logprob, all_accuraries_linear=all_accuraries_linear,
                 all_accuraries_polynomial=all_accuraries_polynomial, all_targets_with_norm=all_targets_with_norm, all_features_with_norm=all_features_with_norm)

def proportion_function_types(mode, first=False):

    # set env_name and color_stats based on mode
    if mode == 0:
        env_name = f'{SYS_PATH}/functionlearning/data/generated_tasks/claude_generated_functionlearningtasks_paramsNA_dim1_data20_tasks9991_run0_procid0_pversion2'
        color_stats = '#405A63'  # '#2F4A5A'# '#173b4f'
    elif mode == 1:  
        env_name = f'{SYS_PATH}/functionlearning/data/generated_tasks/synthetic_functionlearning_tasks_dim1_data25_tasks10000'
        color_stats = '#66828F'  # 5d7684'# '#5d7684'
    elif mode == 2: 
        env_name = f'{SYS_PATH}/functionlearning/data/generated_tasks/real_data'
        color_stats = '#173b4f'  # '#0D2C3D' #'#8b9da7'

    # Load the dataset
    df = pd.read_csv(f'{env_name}.csv')
    max_tasks = 3000 if mode == 0 or mode==1 else 750 #1000
    tasks = range(0, max_tasks) if first else np.random.choice(df.task_id.unique(), max_tasks, replace=False)
    # Initialize a list to store model parameters
    model_params_df = None
    linear_model_params_df = None
    if os.path.exists(f'{SYS_PATH}/functionlearning/data/stats/fitted_model_params_function_types_{str(mode)}.csv'):
        model_params_df = pd.read_csv(f'{SYS_PATH}/functionlearning/data/stats/fitted_model_params_function_types_{str(mode)}.csv')
        linear_model_params_df = pd.read_csv(f'{SYS_PATH}/functionlearning/data/stats/fitted_linear_model_params_function_types_{str(mode)}.csv')
        tasks = model_params_df['task_id'].unique()

    df = df[df['task_id'].isin(tasks)]
    df['input'] = df['input'].apply(lambda x: eval(x)[0]) if mode == 0 or mode ==2 else df['input']
    df['target'] = df['target'].apply(lambda x: eval(x)[0]) if mode == 2 else df['target']
    # max normalize the input and target columns
    df['input'] = df.groupby('task_id')['input'].transform(lambda x: x / x.max())
    df['target'] = df.groupby('task_id')['target'].transform(lambda x: x / x.max())
    # max-min normalize the inputs and targets
    # df['input'] = df.groupby('task_id')['input'].transform(lambda x: (x - x.min())/(x.max() - x.min() + 1e-6))
    # df['target'] = df.groupby('task_id')['target'].transform(lambda x: (x - x.min())/(x.max() - x.min() + 1e-6))
    # standardize the inputs and targets
    # df['input'] = df.groupby('task_id')['input'].transform(lambda x: (x - x.mean())/x.std())
    # df['target'] = df.groupby('task_id')['target'].transform(lambda x: (x - x.mean())/x.std())
    # remove task_ids containing nans for input or target
    

    # Define functions for cubic, quadratic, and exponential models
    # def cubic_model(x, a, b, c, d):
    #     return a * x**3 + b * x**2 + c * x + d

    def periodic_model(x, a, b, c, d):
        return a * np.sin(b * x + c) + d

    def quadratic_model(x, a, b, c):
        return a * x**2 + b * x + c

    def exponential_model(x, a, b, c):
        return a * np.exp(b * x) + c
    
    def linear_model(x, a, b):
        return a * x + b

    # Calculate BIC
    def calculate_bic(y, y_pred, num_params):
        n = len(y)
        mse = mean_squared_error(y, y_pred)
        bic = n * np.log(mse) + num_params * np.log(n)
        return bic


    # Group by task_id and fit models for each task
    if model_params_df is None and linear_model_params_df is None:
        model_params = []
        linear_model_params = []
        for task_id, group in df.groupby('task_id'):
            X = group['input'].values
            y = group['target'].values
            try:
                # Fit linear model
                # linear_model = LinearRegression()
                # linear_model.fit(X.reshape(-1, 1), y)
                # linear_bic = calculate_bic(y, linear_model.predict(X.reshape(-1, 1)), 2)
                # Fit linear model
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        popt_linear, _ = curve_fit(linear_model, X, y)
                        linear_bic = calculate_bic(y, linear_model(X, *popt_linear), len(popt_linear))
                except (ValueError, RuntimeError, OptimizeWarning) as e:
                    print(f"Warning: {e}")
                    linear_bic = float('inf')

                # Fit quadratic model
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        popt_quad, _ = curve_fit(quadratic_model, X, y)
                        quad_bic = calculate_bic(y, quadratic_model(X, *popt_quad), len(popt_quad))
                except (ValueError, RuntimeError, OptimizeWarning) as e:
                    print(f"Warning: {e}")
                    quad_bic = float('inf')

                # Fit cubic model
                # try:
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("error", OptimizeWarning)
                #         popt_cubic, _ = curve_fit(cubic_model, X, y)
                #         cubic_bic = calculate_bic(y, cubic_model(X, *popt_cubic), len(popt_cubic))
                # except (ValueError, RuntimeError, OptimizeWarning) as e:
                #     print(f"Warning: {e}")
                #     cubic_bic = float('inf')

                # Fit exponential model
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        popt_exp, _ = curve_fit(exponential_model, X, y)
                        exp_bic = calculate_bic(y, exponential_model(X, *popt_exp), len(popt_exp))
                except (ValueError, RuntimeError, OptimizeWarning) as e:
                    print(f"Warning: {e}")
                    exp_bic = float('inf')

                # Fit periodic model
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        popt_periodic, _ = curve_fit(periodic_model, X, y)
                        periodic_bic = calculate_bic(y, periodic_model(X, *popt_periodic), len(popt_periodic))
                except (ValueError, RuntimeError, OptimizeWarning) as e:    
                    print(f"Warning: {e}")
                    periodic_bic = float('inf')

                # Select the best model based on BIC
                bics = {'linear': linear_bic,  'quadratic': quad_bic, 'exponential': exp_bic, 'periodic':periodic_bic} #  'cubic': cubic_bic}
                if any([bic == float('inf') for bic in bics.values()]):             ## skip of any bic is inf
                    continue
                best_model = min(bics, key=bics.get)
                if best_model == 'linear':
                    params = {'model': 'linear', 'slope': popt_linear[0], 'intercept': popt_linear[1], 'bic': linear_bic}
                elif best_model == 'quadratic':
                    params = {'model': 'quadratic', 'a': popt_quad[0], 'b': popt_quad[1], 'c': popt_quad[2], 'bic': quad_bic}
                # elif best_model == 'cubic':
                #     params = {'model': 'cubic', 'a': popt_cubic[0], 'b': popt_cubic[1], 'c': popt_cubic[2], 'd': popt_cubic[3], 'bic': cubic_bic}
                elif best_model == 'periodic':
                    params = {'model': 'periodic', 'a': popt_periodic[0], 'b': popt_periodic[1], 'c': popt_periodic[2], 'd': popt_periodic[3], 'bic': periodic_bic}
                elif best_model == 'exponential':
                    params = {'model': 'exponential', 'a': popt_exp[0], 'b': popt_exp[1], 'c': popt_exp[2], 'bic': exp_bic}
                params['task_id'] = task_id
                model_params.append(params)
                linear_model_params.append({'slope': popt_linear[0], 'intercept': popt_linear[1]})

                # Store the model parameters and BIC values for each task
                # if any([bic == float('inf') for bic in bics.values()]):
                #     continue
                # # save all model params and bics
                # linear_params = {'task_id': task_id, 'model': 'linear', 'slope': popt_linear[0], 'intercept': popt_linear[1], 'bic': linear_bic}
                # quad_params = {'task_id': task_id, 'model': 'quadratic', 'a': popt_quad[0], 'b': popt_quad[1], 'c': popt_quad[2], 'bic': quad_bic}
                # # cubic_params = {'task_id': task_id, 'model': 'cubic', 'a': popt_cubic[0], 'b': popt_cubic[1], 'c': popt_cubic[2], 'd': popt_cubic[3], 'bic': cubic_bic}
                # periodic_params = {'task_id': task_id, 'model': 'periodic', 'a': popt_periodic[0], 'b': popt_periodic[1], 'c': popt_periodic[2], 'd': popt_periodic[3], 'bic': periodic_bic}
                # exp_params = {'task_id': task_id, 'model': 'exponential', 'a': popt_exp[0], 'b': popt_exp[1], 'c': popt_exp[2], 'bic': exp_bic}
                # model_params.append(linear_params)
                # model_params.append(quad_params)
                # # model_params.append(cubic_params)
                # model_params.append(periodic_params)
                # model_params.append(exp_params)
                
            except Exception as e:
                print(e)

        # Create a DataFrame from the model parameters
        model_params_df = pd.DataFrame(model_params)
        linear_model_params_df = pd.DataFrame(linear_model_params)

        # save
        if not os.path.exists(f'{SYS_PATH}/functionlearning/data/stats/fitted_model_params_function_types_{str(mode)}.csv'):
            model_params_df.to_csv(f'{SYS_PATH}/functionlearning/data/stats/fitted_model_params_function_types_{str(mode)}.csv', index=False)
            linear_model_params_df.to_csv(f'{SYS_PATH}/functionlearning/data/stats/fitted_linear_model_params_function_types_{str(mode)}.csv', index=False)
    
    # model_bics = model_params_df.groupby('model')['bic'].sum()     # compute total bics for each model
    # model_bics = model_bics.sort_values(ascending=True)     # sort the models based on the total bics
    # fig, ax = plt.subplots(figsize=(10, 6))
    # sns.barplot(x=model_bics.index, y=model_bics.values, palette=['blue', 'green', 'red', 'purple'], ax=ax)
    # sns.despine()
    # ax.set_ylabel('BIC', fontsize=FONTSIZE)
    # ax.set_xlabel('Model', fontsize=FONTSIZE)
    # ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    # plt.grid(visible=False)
    # plt.show()
    # plt.savefig(f'{SYS_PATH}/figures/functionlearning_totalbic_function_types_{str(mode)}.svg', bbox_inches='tight')


    # compute the the number of times bic of each model is the best for each task
    # model_proportions = model_params_df.groupby('model')['bic'].idxmin().value_counts(normalize=True)

    sns.set(style="whitegrid")
    # find number of times a model is in the dataset
    model_proportions = model_params_df['model'].value_counts(normalize=True) 
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=model_proportions.index, y=model_proportions.values, ax=ax)
    sns.despine()
    ax.set_ylabel('Proportion', fontsize=FONTSIZE)
    ax.set_xlabel('Model', fontsize=FONTSIZE)
    # ax.set_title('Proportions of Different Models Being the Best Model', fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    plt.grid(visible=False)
    plt.show()
    plt.savefig(f'{SYS_PATH}/figures/functionlearning_proportion_function_types_{str(mode)}.png', bbox_inches='tight')

    # Create a bar plot for the fitted parameters of the linear model
    linear_params_df = model_params_df[model_params_df['model'] == 'linear'] #linear_model_params_df #
    fig, ax = plt.subplots(figsize=(10, 6))
    proc_df = pd.DataFrame({'Slope': linear_params_df['slope'], 'Intercept': linear_params_df['intercept']})
    sns.barplot(data=proc_df, capsize=.1, errorbar="se", ax=ax)
    sns.despine()
    ax.set_ylabel('Fitted parameters values', fontsize=FONTSIZE)
    ax.set_xlabel('Parameter', fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    # ax.set_title('Fitted Parameters of the Linear Model', fontsize=FONTSIZE)
    plt.grid(visible=False)
    plt.show()
    plt.savefig(f'{SYS_PATH}/figures/functionlearning_fitted_parameters_linear_model_{str(mode)}.png', bbox_inches='tight')

    # find top-3 functions from each model type based on the fitted bics and plot them as line plots
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in ['linear', 'quadratic',  'exponential']:#'cubic',
        top_3 = model_params_df[model_params_df['model'] == model].sort_values('bic').head(5)
        for i, row in top_3.iterrows():
            task_data = df[df['task_id'] == row['task_id']]
            sns.lineplot(x=task_data['input'], y=task_data['target'], ax=ax)
        sns.despine()
        ax.set_ylabel('Target', fontsize=FONTSIZE)
        ax.set_xlabel('Input', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.title('Sampled functions', fontsize=FONTSIZE)
    sns.despine()
    plt.grid(visible=False)
    plt.show()
    plt.savefig(f'{SYS_PATH}/figures/functionlearning_top5functionspermodel_{str(mode)}.png', bbox_inches='tight')

def model_errors_function_types(FIGSIZE=(12, 6)):
    # Load the data
    results_mi = np.load(f'{PARADIGM_PATH}/data/model_simulation/task=syntheticfunctionlearning_experiment=1_source=synthetic_condition=unknown_loss=nll_paired=False_policy=greedy_ess=0.0.npz')
    results_ermi = np.load(f'{PARADIGM_PATH}/data/model_simulation/task=syntheticfunctionlearning_experiment=1_source=claude_condition=unknown_loss=nll_paired=False_policy=greedy_ess=0.0.npz')

    # Extract unique functions and calculate MSE for results_ermi
    functions = ['positive_linear', 'negative_linear', 'quadratic', 'radial_basis']
    function_names = {'positive_linear': 'Positive Linear', 'negative_linear': 'Negative Linear', 'quadratic': 'Quadratic', 'radial_basis': 'Radial Basis'}
    error_dict_ermi = {'Function': [], 'MSE': [], 'Dataset': [], 'Per_trial_MSE': []}
    error_dict_mi = {'Function': [], 'MSE': [], 'Dataset': [], 'Per_trial_MSE': []}
    for function in functions:
        mse = results_ermi['model_errors'].squeeze()[(results_ermi['ground_truth_functions'] == function)].mean()
        num_trials = results_ermi['per_trial_model_errors'].shape[-1]
        ground_truth_functions_repeated = np.repeat(results_ermi['ground_truth_functions'][:, :, np.newaxis], num_trials, axis=2).reshape(-1, num_trials)
        per_trial_mse = results_ermi['per_trial_model_errors'].reshape(-1, num_trials)[(ground_truth_functions_repeated == function)].reshape(-1, num_trials)
        error_dict_ermi['Function'].append(function_names[function])
        error_dict_ermi['MSE'].append(mse)
        error_dict_ermi['Dataset'].append('ERMI')
        error_dict_ermi['Per_trial_MSE'].append(per_trial_mse)
        
        mse = results_mi['model_errors'].squeeze()[(results_mi['ground_truth_functions'] == function)].mean()
        per_trial_mse = results_mi['per_trial_model_errors'].reshape(-1, num_trials)[(ground_truth_functions_repeated == function)].reshape(-1, num_trials)
        error_dict_mi['Function'].append(function_names[function])
        error_dict_mi['MSE'].append(mse)
        error_dict_mi['Dataset'].append('MI')
        error_dict_mi['Per_trial_MSE'].append(per_trial_mse)

    # Combine the data into a single DataFrame
    df_ermi = pd.DataFrame(error_dict_ermi)
    df_mi = pd.DataFrame(error_dict_mi)
    df_combined = pd.concat([df_ermi, df_mi])

    # Plot the combined data
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=df_combined, x='Function', y='MSE', hue='Dataset', capsize=.1, errorbar="sd", ax=ax)
    sns.despine()
    ax.legend(frameon=False, fontsize=FONTSIZE-2)
    ax.set_ylabel('Mean-squared Error', fontsize=FONTSIZE)
    ax.set_xlabel('Function', fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    plt.grid(visible=False)
    plt.show()
    plt.savefig(f'{SYS_PATH}/figures/functionlearning_model_error_function_types.png', bbox_inches='tight')

    # Plot the per-trial MSE
    sns.set(style="whitegrid")
    for dataset in df_combined['Dataset'].unique():
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for function in df_combined['Function'].unique():
            subset = df_combined[(df_combined['Function'] == function) & (df_combined['Dataset'] == dataset)]
            per_trial_mse = np.array(subset['Per_trial_MSE'].values[0])
            ax.plot(per_trial_mse.mean(axis=0), label=f'{function}', lw=2)
        ax.set_xlabel('Trial', fontsize=FONTSIZE)
        ax.set_ylabel('MSE', fontsize=FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
        ax.legend(frameon=False, fontsize=FONTSIZE-2)
        plt.grid(visible=False)
        sns.despine()
        plt.show()
        plt.savefig(f'{SYS_PATH}/figures/functionlearning_per_trial_mse_function_types_{dataset}.png', bbox_inches='tight')

def model_extrapolation_delosh1997(FIGSIZE=(12, 6)):
    # load model
    results_ermi = np.load(f'{PARADIGM_PATH}/data/model_simulation/task=delosh1997_experiment=1_source=claude_condition=unknown_loss=nll_paired=False_policy=greedy_ess=0.0.npz')
    results_mi = np.load(f'{PARADIGM_PATH}/data/model_simulation/task=delosh1997_experiment=1_source=synthetic_condition=unknown_loss=nll_paired=False_policy=greedy_ess=0.0.npz')

    # Extract unique functions and calculate MSE for results_ermi
    functions = ['linear', 'exponential', 'quadratic']
    function_names = {'linear': 'Linear', 'exponential': 'Exponential', 'quadratic': 'Quadratic'}
    error_dict_ermi = {'Function': [], 'MSE': [], 'Dataset': [], 'Input': [], 'Target': [], 'Extrapolation_Input': [], 'Extrapolation_Target': [], 'Per_trial_MSE': []}
    error_dict_mi = {'Function': [], 'MSE': [], 'Dataset': [], 'Input': [], 'Target': [], 'Extrapolation_Input': [], 'Extrapolation_Target': [], 'Per_trial_MSE': []}
    for function in functions:
        mse = results_ermi['model_errors'].squeeze()[(results_ermi['ground_truth_functions'] == function)].mean()
        input_data = results_ermi['human_preds'].squeeze()[(results_ermi['ground_truth_functions'] == function)][:, :-1]
        target_data = results_ermi['targets'].squeeze()[(results_ermi['ground_truth_functions'] == function)][:, :-1]
        extrapolation_input = results_ermi['human_preds'].squeeze()[(results_ermi['ground_truth_functions'] == function)][:, -1]
        extrapolation_target = results_ermi['model_preds'].squeeze()[(results_ermi['ground_truth_functions'] == function)][:, -1]
        num_trials = results_ermi['per_trial_model_errors'].shape[-1]
        ground_truth_functions_repeated = np.repeat(results_ermi['ground_truth_functions'][:, :, np.newaxis], num_trials, axis=2).reshape(-1, num_trials)
        per_trial_mse = results_ermi['per_trial_model_errors'].reshape(-1, num_trials)[(ground_truth_functions_repeated == function)].reshape(-1, num_trials)
        error_dict_ermi['Function'].append(function_names[function])
        error_dict_ermi['MSE'].append(mse)
        error_dict_ermi['Dataset'].append('ERMI')
        error_dict_ermi['Input'].append(input_data)
        error_dict_ermi['Target'].append(target_data)
        error_dict_ermi['Extrapolation_Input'].append(extrapolation_input)
        error_dict_ermi['Extrapolation_Target'].append(extrapolation_target)
        error_dict_ermi['Per_trial_MSE'].append(per_trial_mse)
        
        mse = results_mi['model_errors'].squeeze()[(results_mi['ground_truth_functions'] == function)].mean()
        input_data = results_mi['human_preds'].squeeze()[(results_mi['ground_truth_functions'] == function)][:, :-1]
        target_data = results_mi['targets'].squeeze()[(results_mi['ground_truth_functions'] == function)][:, :-1]
        extrapolation_input = results_mi['human_preds'].squeeze()[(results_mi['ground_truth_functions'] == function)][:, -1]
        extrapolation_target = results_mi['model_preds'].squeeze()[(results_mi['ground_truth_functions'] == function)][:, -1]
        per_trial_mse = results_mi['per_trial_model_errors'].reshape(-1, num_trials)[(ground_truth_functions_repeated == function)].reshape(-1, num_trials)
        error_dict_mi['Function'].append(function_names[function])
        error_dict_mi['MSE'].append(mse)
        error_dict_mi['Dataset'].append('MI')
        error_dict_mi['Input'].append(input_data)
        error_dict_mi['Target'].append(target_data)
        error_dict_mi['Extrapolation_Input'].append(extrapolation_input)
        error_dict_mi['Extrapolation_Target'].append(extrapolation_target)
        error_dict_mi['Per_trial_MSE'].append(per_trial_mse)

    # Combine the data into a single DataFrame
    df_ermi = pd.DataFrame(error_dict_ermi)
    df_mi = pd.DataFrame(error_dict_mi)
    df_combined = pd.concat([df_ermi, df_mi])

    # scatter plot of Input vs Target and Extrapolation Input vs Extrapolation Target for each function
    sns.set(style="whitegrid")
    for dataset in df_combined['Dataset'].unique():
        for function in df_combined['Function'].unique():
            subset = df_combined[(df_combined['Function'] == function) & (df_combined['Dataset'] == dataset)]
            fig, axs = plt.subplots(1, 1, figsize=(6, 6))
            for i, row in subset.iterrows():
                axs.scatter(row['Extrapolation_Input'], row['Extrapolation_Target'], label=f'Test', alpha=0.5, color='red')
                axs.scatter(row['Input'], row['Target'], label='Training', alpha=0.5, color='black')
            axs.set_xlabel('Input', fontsize=FONTSIZE)
            axs.set_ylabel('Target', fontsize=FONTSIZE)
            axs.legend(frameon=False, fontsize=FONTSIZE-2)
            axs.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
            # add vertical dotted line as a separator at x = -0.5 and x = 0.5
            axs.axvline(x=-0.25, color='black', linestyle='--', alpha=0.5)
            axs.axvline(x=0.25, color='black', linestyle='--', alpha=0.5)
            plt.grid(visible=False)
            sns.despine()
            plt.show()
            plt.savefig(f'{SYS_PATH}/figures/functionlearning_extrapolation_{function}_{dataset}.png', bbox_inches='tight')

    # Plot the per-trial MSE
    sns.set(style="whitegrid")
    for dataset in df_combined['Dataset'].unique():
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for function in df_combined['Function'].unique():
            subset = df_combined[(df_combined['Function'] == function) & (df_combined['Dataset'] == dataset)]
            per_trial_mse = np.array(subset['Per_trial_MSE'].values[0])
            ax.plot(per_trial_mse.mean(axis=0), label=f'{function}', lw=2)
        ax.set_xlabel('Trial', fontsize=FONTSIZE)
        ax.set_ylabel('MSE', fontsize=FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
        ax.legend(frameon=False, fontsize=FONTSIZE-2)
        plt.grid(visible=False)
        sns.despine()
        plt.show()
        plt.savefig(f'{SYS_PATH}/figures/functionlearning_extrapolation_per_trial_mse_function_types_{dataset}.png', bbox_inches='tight')
    
    # Plot the combined data
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=df_combined, x='Function', y='MSE', hue='Dataset', capsize=.1, errorbar="sd", ax=ax)
    sns.despine()
    ax.legend(frameon=False, fontsize=FONTSIZE-2)
    ax.set_ylabel('Mean-squared Error', fontsize=FONTSIZE)
    ax.set_xlabel('Function', fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    plt.grid(visible=False)
    plt.show()
    plt.savefig(f'{SYS_PATH}/figures/functionlearning_extrapolation_mse_function_types.png', bbox_inches='tight')

        
def model_comparison_little2024(FIGSIZE=(5,5)):
    sns.set(style="whitegrid")
    task_name = 'little2022'
    ess = 0.0
    mi = np.load(f'{PARADIGM_PATH}/data/model_comparison/task=little2022_experiment=1_source=synthetic_condition=unknown_loss=nll_paired=False_method=unbounded.npz')
    ermi = np.load(f'{PARADIGM_PATH}/data/model_comparison/task=little2022_experiment=1_source=claude_condition=unknown_loss=nll_paired=False_method=unbounded.npz')
    bermi = np.load(f'{PARADIGM_PATH}/data/model_comparison/task=little2022_experiment=1_source=claude_condition=unknown_loss=nll_paired=False_method=bounded.npz')
    bmi = np.load(f'{PARADIGM_PATH}/data/model_comparison/task=little2022_experiment=1_source=synthetic_condition=unknown_loss=nll_paired=False_method=bounded.npz')
    ref = np.load(f'{PARADIGM_PATH}/data/model_simulation/env=synthetic_dim1_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.01_shuffleTrue_run=0_synthetic.npz')

    df = pd.DataFrame.from_dict({
                                'BERMI':bermi['nlls'],
                                'BMI':bmi['nlls'],
                                'ERMI':ermi['nlls'],
                                'MI':mi['nlls'],
                                })

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=df, capsize=.1, errorbar="sd", ax=ax)
    sns.swarmplot(data=df, color="0", alpha=.35, ax=ax)
    sns.despine()
    ax.set_ylabel('Mean-squared Error', fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    plt.grid(visible=False)
    plt.show()
    plt.savefig(f'{SYS_PATH}/figures/functionlearning_model_comparison_little2024.png', bbox_inches='tight')

    results_bermi = np.load(f'{PARADIGM_PATH}/data/model_simulation/task={task_name}_experiment=1_source=claude_condition=unknown_loss=nll_paired=False_policy=greedy_ess={str(bermi["ess"][np.argmin(bermi["nlls"])])}.npz')
    results_bmi = np.load(f'{PARADIGM_PATH}/data/model_simulation/task={task_name}_experiment=1_source=synthetic_condition=unknown_loss=nll_paired=False_policy=greedy_ess={str(bmi["ess"][np.argmin(bmi["nlls"])])}.npz')
    results_ermi = np.load(f'{PARADIGM_PATH}/data/model_simulation/task={task_name}_experiment=1_source=claude_condition=unknown_loss=nll_paired=False_policy=greedy_ess=0.0.npz')
    results_mi = np.load(f'{PARADIGM_PATH}/data/model_simulation/task={task_name}_experiment=1_source=synthetic_condition=unknown_loss=nll_paired=False_policy=greedy_ess=0.0.npz')
    models = [results_bermi, results_bmi, results_ermi, results_mi]
    subjects =  [np.argmin(bermi['nlls']), np.argmin(bmi['nlls']), np.argmin(ermi['nlls']), np.argmin(mi['nlls'])] # best parrticipant for each model
    model_names = ['BERMI', 'BMI', 'ERMI', 'MI']

    num_functions = ref['human_preds'].shape[1]
    num_participants = ref['model_preds'].shape[0]
    num_data = 24
    for (subject, model, model_name) in zip(subjects, models, model_names):
        sns.set(style="whitegrid")
        fig, axs = plt.subplots(1, num_functions, figsize=(6*num_functions, 4))
        for function in range(num_functions):
            axs[function].plot(model['human_preds'][subject, function, :, 0], model['model_preds'].reshape(num_participants, num_functions, num_data)[subject, function], lw=2, label=model_name)
            axs[function].scatter(ref['ground_truth_functions'][subject, function, :, 0], ref['ground_truth_functions'][subject, function, :, 1], c='black', label="Ground Truth")
            axs[function].plot(ref['human_preds'][subject, function, :, 0], ref['human_preds'][subject, function, :, 1], c='green', lw=2, label='Human')
            
            if function == 0:
                axs[function].set_xlabel('Input', fontsize=FONTSIZE)
                axs[function].set_ylabel('Target', fontsize=FONTSIZE)
                axs[function].legend(frameon=False, fontsize=FONTSIZE-4)
            axs[function].tick_params(axis='both', which='major', labelsize=FONTSIZE-4)
            axs[function].grid(visible=False)
            sns.despine()
            plt.tight_layout()
            plt.show()
            plt.savefig(f'{SYS_PATH}/figures/functionlearning_model_comparison_little2024_functions_subject{subject}_model{model_name}.png', bbox_inches='tight')