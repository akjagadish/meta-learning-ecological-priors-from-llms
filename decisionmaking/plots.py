from wordcloud import WordCloud
from collections import Counter
# import torch.nn.functional as F
from groupBMC.groupBMC import GroupBMC
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import get_cmap
import seaborn as sns
import sys
import os
from os import getenv
from os.path import join
from dotenv import load_dotenv
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
load_dotenv()
SYS_PATH = getenv('BERMI_DIR')
PARADIGM_PATH = f"{SYS_PATH}/decisionmaking"
sys.path.append(PARADIGM_PATH)
sys.path.append(f"{PARADIGM_PATH}/data")
FONTSIZE = 20

def gini_compute(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    return 0.5 * rmad

def plot_decisionmaking_data_statistics(mode=0, dim=4, condition='unkown', method='best'):

    def calculate_bic(n, rss, num_params):
        bic = n * np.log(rss/n) + num_params * np.log(n)
        # if bic is an array return item else return bic
        return bic.item() if isinstance(bic, np.ndarray) else bic

    def return_data_stats(data, poly_degree=2, first=False, dim=dim, include_bias=True):

        df = data.copy()
        max_tasks = 400
        max_trial = 20
        # sample tasks
        tasks = range(0, max_tasks) if first else np.random.choice(df.task_id.unique(), max_tasks, replace=False)
        all_corr, all_bics_linear, all_bics_quadratic, gini_coeff, all_accuraries_linear, all_accuraries_polynomial = [], [], [], [], [], []
        sign_coeff, direction_coeff = [], []
        all_features_without_norm, all_features_with_norm, all_targets_with_norm = np.array(
            []), np.array([]), np.array([])
        for i in tasks:
            df_task = df[df['task_id'] == i]
            if len(df_task) > 0:  # arbitary data size threshold
                y = df_task['target'].to_numpy()
                X = df_task["input"].to_numpy()
                X = np.stack(X)
                X = (X - X.mean(axis=0))/(X.std(axis=0) + 1e-6)
                y = (y - y.mean(axis=0))/(y.std(axis=0) + 1e-6)

                # all_corr.append(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
                # all_corr.append(np.corrcoef(X[:, 0], X[:, 2])[0, 1])
                # all_corr.append(np.corrcoef(X[:, 0], X[:, 3])[0, 1])
                # all_corr.append(np.corrcoef(X[:, 1], X[:, 2])[0, 1])
                # all_corr.append(np.corrcoef(X[:, 1], X[:, 3])[0, 1])
                # all_corr.append(np.corrcoef(X[:, 2], X[:, 3])[0, 1])
                all_corr.append(np.corrcoef(X.T)[np.triu_indices(dim, k=1)])

                all_features_with_norm = np.concatenate(
                    [all_features_with_norm, X.flatten()])
                all_targets_with_norm = np.concatenate(
                    [all_targets_with_norm, y.flatten()])

                if (y == 0).all() or (y == 1).all():
                    pass
                else:
                    # X_linear = PolynomialFeatures(1).fit_transform(X)
                    X_linear = PolynomialFeatures(
                        1, include_bias=include_bias).fit_transform(X)

                    # # linear regression from X_linear to y
                    # linear_regresion = sm.OLS(y, X_linear).fit()

                    # # polinomial regression from X_poly
                    # X_poly = PolynomialFeatures(
                    #     poly_degree, interaction_only=True, include_bias=False).fit_transform(X)
                    # polynomial_regression = sm.OLS(y, X_poly).fit()

                    # gini coefficient from linear regression coefficients
                    params = sm.OLS(y, X_linear).fit().params
                    gini_coeff.append(gini_compute(
                        np.abs(params[1 if include_bias else 0:])))

                    per_feature_params = np.zeros((dim))
                    for i in range(dim):
                        per_feature_params[i] = sm.OLS(
                            y, X_linear[:, [0, i+1]]).fit().params[1 if include_bias else 0]

                    # sign of the coefficients
                    sign_coeff.append(np.sign(per_feature_params))

                    # direction of the coefficients
                    direction_coeff.append(per_feature_params)

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

        return all_corr, gini_coeff, posterior_logprob, all_accuraries_linear, all_accuraries_polynomial, all_targets_with_norm, all_features_with_norm, sign_coeff, direction_coeff

    # set env_name and color_stats based on mode
    if mode == 0: # claude
        if dim == 2:
            env_name = f'{SYS_PATH}/decisionmaking/data/claude_generated_functionlearningtasks_paramsNA_dim2_data20_tasks9254_run0_procid0_pversion2_{condition}'
        elif dim == 4:
            if 'pseudo' in condition:
                assert condition in ['pseudoranked', 'pseudodirection'], 'condition must be ranked or direction'
                env_name = f'{SYS_PATH}/decisionmaking/data/claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks7284_run0_procid1_pversionunknown_{condition}'               
            else:
                num_tasks = 8770 if condition == 'ranked' else 8220 if condition == 'direction' else 7284
                env_name = f'{SYS_PATH}/decisionmaking/data/claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks{num_tasks}_run0_procid1_pversion{condition}'         
        color_stats = '#405A63'  # '#2F4A5A'# '#173b4f'
    elif mode == 1:  # synthetic
        if dim==2:
            env_name = f'{SYS_PATH}/decisionmaking/data/synthetic_decisionmaking_tasks_dim2_data20_tasks10000'
        elif dim==4:
            env_name = f'{SYS_PATH}/decisionmaking/data/synthetic_decisionmaking_tasks_dim4_data20_tasks400_{condition}'
        color_stats = '#66828F'  # 5d7684'# '#5d7684'
    elif mode == 2:  # real
        assert (condition == 'openML') or (condition == 'lichtenberg2017'), 'condition must be openML or lichtenberg2017'
        env_name = f'{SYS_PATH}/decisionmaking/data/real_data_dim{dim}_method{method}_{condition}'
        color_stats = '#173b4f'  # '#0D2C3D' #'#8b9da7'
    # elif mode == 3:
    #     env_name = f'{SYS_PATH}/decisionmaking/data/synthetic_tasks_dim4_data650_tasks1000_nonlinearTrue'
    #     color_stats = '#5d7684'

    # load data
    data = pd.read_csv(f'{env_name}.csv')
    data.input = data['input'].apply(lambda x: np.array(eval(x)))
    if mode == 2: # or mode == 1:
        data.target = data['target'].apply(lambda x: np.array(eval(x)))
        # TODO: shuffle order of input features (but it is artifiically inducing lack of ranking)
        data.input = data.input.apply(np.random.permutation)

    if os.path.exists(f'{SYS_PATH}/decisionmaking/data/stats/stats_{str(mode)}_{str(dim)}_{condition}.npz'):
        stats = np.load(
            f'{SYS_PATH}/decisionmaking/data/stats/stats_{str(mode)}_{str(dim)}_{condition}.npz', allow_pickle=True)
        all_corr, gini_coeff, posterior_logprob, all_accuraries_linear = stats['all_corr'], stats[
            'gini_coeff'], stats['posterior_logprob'], stats['all_accuraries_linear']
        all_accuraries_polynomial, all_targets_with_norm, all_features_with_norm, sign_coeff, direction_coeff = stats[
            'all_accuraries_polynomial'], stats['all_targets_with_norm'], stats['all_features_with_norm'], stats['sign_coeff'], stats['direction_coeff']
    else:
        all_corr, gini_coeff, posterior_logprob, all_accuraries_linear, all_accuraries_polynomial, \
            all_targets_with_norm, all_features_with_norm, sign_coeff, direction_coeff = return_data_stats(
                data, dim=dim)

    gini_coeff = np.array(gini_coeff)
    gini_coeff = gini_coeff[~np.isnan(gini_coeff)]
    all_corr = np.array(all_corr)
    sign_coeff = np.array(sign_coeff)
    direction_coeff = np.stack(direction_coeff)
    # posterior_logprob = posterior_logprob[:, 0].exp().detach().numpy()

    FONTSIZE = 22  # 8
    fig, axs = plt.subplots(1, 4,  figsize=(6*4, 4))  # figsize=(6.75, 1.5))
    sns.histplot(all_corr.reshape(-1), ax=axs[0], bins=11, binrange=(
        -1., 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    sns.histplot(gini_coeff, ax=axs[1], bins=11, binrange=(
        0., gini_coeff.max()), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    sns.histplot(np.argmax(np.abs(direction_coeff), axis=1), ax=axs[2], bins=dim, binrange=(
        -0.5, dim-0.5), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    sns.histplot(sign_coeff.reshape(-1), ax=axs[3], bins=3, binrange=(
        -1.5, 1.5), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)

    axs[0].set_ylim(0, .4)
    axs[1].set_ylim(0, .4)
    # axs[2].set_ylim(0, .6)
    axs[3].set_ylim(0, 1.)
    # axs[3].set_ylim(0,  0.75)

    # axs[0].set_yticks(np.arange(0.5, 1.05, 0.25))
    # axs[1].set_yticks(np.arange(0, 0.45, 0.2))
    # axs[2].set_yticks(np.arange(0, 0.4, 0.15))
    # axs[3].set_yticks(np.arange(0, 1.05, 0.5))

    axs[2].set_xticks(np.arange(0, dim, 1))
    axs[2].set_xticklabels([f"coef{i+1}" for i in range(dim)])
    axs[3].set_xticks(np.arange(-1, 2, 1))
    axs[3].set_xticklabels(['negative', 'unsigned', 'positive'])

    # set tick size
    axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    axs[3].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)

    axs[0].set_ylabel('Proportion', fontsize=FONTSIZE)
    axs[1].set_ylabel('', fontsize=FONTSIZE)
    axs[2].set_ylabel('')
    axs[3].set_ylabel('')

    if mode == 3:
        axs[0].set_xlabel('Pearson\'s r', fontsize=FONTSIZE)
        axs[1].set_xlabel('Gini coefficient', fontsize=FONTSIZE)
        axs[2].set_xlabel('Regression coefficient', fontsize=FONTSIZE)
        axs[3].set_xlabel('Sign of regression coefficient', fontsize=FONTSIZE)

    # set title
    if mode == 2:
        axs[0].set_title('Input Correlation', fontsize=FONTSIZE)
        axs[1].set_title('Sparsity', fontsize=FONTSIZE)
        axs[2].set_title('Ranking', fontsize=FONTSIZE)
        axs[3].set_title('Direction', fontsize=FONTSIZE)

    plt.tight_layout()
    sns.despine()
    plt.savefig(
        f'{SYS_PATH}/figures/decisionmaking_stats_{str(mode)}_{str(dim)}_{condition}.svg', bbox_inches='tight')
    plt.show()

    # FONTSIZE = 22  # 8
    # fig, axs = plt.subplots(1, 4,  figsize=(6*4, 4))  # figsize=(6.75, 1.5))
    # axs[0].plot(all_accuraries_linear, color=color_stats, alpha=1., lw=3)
    # # axs[0].plot(all_accuraries_polynomial, alpha=0.7)
    # sns.histplot(np.stack(all_corr), ax=axs[1], bins=11, binrange=(
    #     0., 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    # sns.histplot(np.stack(all_targets_with_norm).reshape(-1), ax=axs[2], bins=11, binrange=(
    #     0., 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    # sns.histplot(posterior_logprob, ax=axs[3], bins=5, binrange=(
    #     0.0, 1.), stat='probability', edgecolor='w', linewidth=1, color=color_stats, alpha=1.)
    # # axs[1].set_xlim(-1, 1)

    # axs[0].set_ylim(0., 1.05)
    # axs[1].set_ylim(0,  0.2)
    # axs[2].set_ylim(0,  0.2)
    # axs[2].set_xlim(0., 1.05)
    # axs[3].set_xlim(0., 1.05)

    # # axs[0].set_yticks(np.arange(0.5, 1.05, 0.25))
    # # axs[1].set_yticks(np.arange(0, 0.45, 0.2))
    # # axs[2].set_yticks(np.arange(0, 0.4, 0.15))
    # axs[3].set_yticks(np.arange(0, 1.05, 0.5))

    # # set tick size
    # axs[0].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    # axs[1].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    # axs[2].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)
    # axs[3].tick_params(axis='both', which='major', labelsize=FONTSIZE-2)

    # axs[0].set_ylabel('Squared Error', fontsize=FONTSIZE)
    # axs[1].set_ylabel('Proportion', fontsize=FONTSIZE)
    # axs[2].set_ylabel('')
    # axs[3].set_ylabel('')

    # if mode == 3:
    #     axs[0].set_xlabel('Trials', fontsize=FONTSIZE)
    #     axs[1].set_xlabel('Input', fontsize=FONTSIZE)
    #     axs[2].set_xlabel('Target', fontsize=FONTSIZE)
    #     axs[3].set_xlabel('Posterior probability ', fontsize=FONTSIZE)

    # # set title
    # if mode == 2:
    #     axs[0].set_title('Performance', fontsize=FONTSIZE)
    #     axs[1].set_title('Input distribution', fontsize=FONTSIZE)
    #     axs[2].set_title('Target distribution', fontsize=FONTSIZE)
    #     axs[3].set_title('Linearity', fontsize=FONTSIZE)

    # plt.tight_layout()
    # sns.despine()
    # plt.savefig(
    #     f'{SYS_PATH}/figures/supp_decisionmaking_stats_{str(mode)}_{str(dim)}_{condition}_test.svg', bbox_inches='tight')
    # plt.show()

    # save computed stats in one .npz file
    if not os.path.exists(f'{SYS_PATH}/decisionmaking/data/stats/stats_{str(mode)}_{str(dim)}_{condition}.npz'):
        np.savez(f'{SYS_PATH}/decisionmaking/data/stats/stats_{str(mode)}_{str(dim)}_{condition}.npz', all_corr=all_corr, gini_coeff=gini_coeff, posterior_logprob=posterior_logprob, all_accuraries_linear=all_accuraries_linear,
                 all_accuraries_polynomial=all_accuraries_polynomial, all_targets_with_norm=all_targets_with_norm, all_features_with_norm=all_features_with_norm, sign_coeff=sign_coeff, direction_coeff=direction_coeff)

def world_cloud(file_name, path='/u/ajagadish/ermi/decisionmaking/data/synthesize_problems', feature_names=True, pairs=False, top_labels=50):

    df = pd.read_csv(f'{path}/{file_name}.csv')
    dim = int(file_name.split("_dim")[1].split("_")[0])
    df.feature_names = df['feature_names'].apply(lambda x: list(eval(x)[:dim]))

    def to_lower(ff):
        return [x.lower() for x in ff]

    # name of the column containing the feature names
    column_name = 'feature_names' if feature_names else 'target_names'
    # count of number of times a type of features occurs
    list_counts = Counter([tuple(features) for features in df[column_name]]
                          if pairs else np.stack(df[column_name].values).reshape(-1))

    # sort the Counter by counts in descending order
    sorted_list_counts = sorted(
        list_counts.items(), key=lambda x: x[1], reverse=True)

    # extract the counts and names for the top 50 labels
    task_labels = np.array([task_label[0]
                            for task_label in sorted_list_counts[:top_labels]])
    label_counts = np.array([task_label[1]
                             for task_label in sorted_list_counts[:top_labels]])
    label_names = ['-'.join(task_labels[idx])
                   for idx in range(len(task_labels))] if pairs else task_labels

    # make a dict with task labels and counts
    word_freq = {}
    for idx in range(len(label_names)):
        word_freq[label_names[idx]] = label_counts[idx]

    # generate word cloud
    # wordcloud = WordCloud(width=800, height=400, max_words=50, background_color='white').generate_from_frequencies(word_freq)
    wordcloud = WordCloud(width=1300, height=700, background_color='white', max_font_size=100,
                          collocations=False, colormap='inferno', prefer_horizontal=1).generate_from_frequencies(word_freq)
    plt.figure(figsize=(13, 7), dpi=1000)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    wordcloud.to_file(
        f'{SYS_PATH}/figures/wordcloud_{column_name}_paired={pairs}_top{top_labels}.png')

def model_simulation_binz2022(experiment_id, source='claude', policy='greedy', condition='unknown', FIGSIZE = (8, 5)):
    
    esses = [0.0, 0.5, 1., 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    dim = 'dim2' if experiment_id == 3 else 'dim4'
    # cond = 'ranking_' if condition == 'ranked' else 'direction_' if condition == 'direction' else ''
    data = pd.read_csv(f'{PARADIGM_PATH}/data/human/binz2022heuristics_exp{experiment_id}.csv')
    num_tasks = data.task.max() + 1
    num_trials = (data.trial.max()+1)
    
    # performance of ERMI and BERMI with different ess values over trials
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ess_list, mean_accs, norms_list  = [], [], []
    cmap = get_cmap('cividis_r') # Generate colors from a colormap
    colors = [cmap(i) for i in np.linspace(0., 1., len(esses)+1)]  # Aajust the number of colors as needed
    for i, ess in enumerate(esses):
        results_bermi_paired_ess = np.load(f'{PARADIGM_PATH}/data/model_simulation/env={source}_dim4_{condition}_model=transformer_num_episodes1000000_num_hidden=8_lr0.0003_num_layers=2_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_ess{str(float(ess))}_std0.1_run=0_essinit0.0_annealed_schedulefree_binz2022.npz')    
        # results_bermi_paired_ess = np.load(f'{PARADIGM_PATH}/data/model_simulation/env={source}_{dim}_{condition}_model=transformer_num_episodes1000000_num_hidden=8_lr0.0003_num_layers=2_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_ess{str(float(ess))}_std0.1_run=0_{cond}essinit0.0_annealed_schedulefree_binz2022.npz')    
        ax.errorbar(x=np.arange(10), y=(results_bermi_paired_ess['per_trial_model_accuracy'] / num_tasks).mean(0),
                    yerr=(results_bermi_paired_ess['per_trial_model_accuracy'] / num_tasks).std(0) / np.sqrt(num_tasks),
                    label=f'{"BERMI" if source == "claude" else "BMI"} $\lambda={str(ess)}$', c=colors[i + 1])
        mean_accs.append((results_bermi_paired_ess['per_trial_model_accuracy'] / num_tasks).mean())
        ess_list.append(ess)
        norms_list.append(results_bermi_paired_ess['l2_norms'].mean())
    ax.set_xlabel('trials', fontsize=FONTSIZE)
    ax.set_ylabel('accuracy', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE - 2)
    plt.yticks(fontsize=FONTSIZE - 2)
    plt.ylim(0.4, 1.0)
    plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0.05), frameon=False, fontsize=FONTSIZE - 10)
    sns.despine()
    f.tight_layout()
    plt.show()
    plt.savefig(f'{SYS_PATH}/figures/binz2022_performance_over_trials_{source}_{condition}_exp{experiment_id}.png')

    # scatter plot of mean accuracy vs regularization parameter
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.scatter(ess_list, mean_accs, lw=3, color=colors[1:])
    ax.set_ylabel('accuracy', fontsize=FONTSIZE)
    ax.set_xlabel('$\lambda$', fontsize=FONTSIZE)
    # ax.set_xscale('log')
    plt.xticks(fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.ylim(0.45, 1.) # set y axis limit between 0.5 and 1.
    sns.despine()
    plt.show()
    plt.savefig(f'{SYS_PATH}/figures/binz2022_meanperformance_vs_ess_{source}_{condition}_exp{experiment_id}.png')

    # scatter plot of mean accuracy vs l2 norm
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.scatter(norms_list, mean_accs, lw=3, color=colors[1:])
    ax.set_ylabel('accuracy', fontsize=FONTSIZE)
    ax.set_xlabel('$norm$', fontsize=FONTSIZE)
    # ax.set_xscale('log')
    plt.xticks(fontsize=FONTSIZE-3)
    plt.yticks(fontsize=FONTSIZE-3)
    plt.ylim(0.45, 1.) # set y axis limit between 0.5 and 1.
    sns.despine()
    plt.show()
    plt.savefig(f'{SYS_PATH}/figures/binz2022_meanperformance_vs_norm_{source}_{condition}_exp{experiment_id}.png')
    
    ginis_bermi =np.zeros((len(esses),) + results_bermi_paired_ess['model_coefficients'][...,[0]].shape)
    for i, ess in enumerate(esses):
        results_bermi_paired_ess = np.load(f'{PARADIGM_PATH}/data/model_simulation/env={source}_{dim}_{condition}_model=transformer_num_episodes1000000_num_hidden=8_lr0.0003_num_layers=2_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_ess{str(float(ess))}_std0.1_run=0_essinit0.0_annealed_schedulefree_binz2022.npz')
        # results_bermi_paired_ess = np.load(f'{PARADIGM_PATH}/data/model_simulation/env={source}_{dim}_{condition}_model=transformer_num_episodes1000000_num_hidden=8_lr0.0003_num_layers=2_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_ess{str(float(ess))}_std0.1_run=0_{cond}essinit0.0_annealed_schedulefree_binz2022.npz')
        for participant in range(results_bermi_paired_ess['model_coefficients'].shape[0]):
            for task in range(results_bermi_paired_ess['model_coefficients'].shape[1]):
                for trial in range(results_bermi_paired_ess['model_coefficients'].shape[2]):
                    ginis_bermi[i, participant, task, trial]= gini_compute(np.abs(results_bermi_paired_ess['model_coefficients'][participant, task, trial]))
                    
    for idx, ess in enumerate(esses):
        all_trials_data = []
        for trial in range(1, num_trials, 2):
            trial_data = ginis_bermi[idx][:, :, trial].squeeze().mean(0)
            all_trials_data.append(trial_data)
        df = pd.DataFrame({'gini': [item for sublist in all_trials_data for item in sublist],
            'trial': [trial*2+1 for trial, sublist in enumerate(all_trials_data) for _ in sublist]})
        f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        sns.swarmplot(x='trial', y='gini', data=df, ax=ax, color=colors[idx + 1])
        plt.ylim([0., 0.5 if experiment_id==3 else 0.8])
        plt.ylabel('gini')
        plt.xlabel('trials')
        sns.despine()
        plt.title(f"experiment {experiment_id}: {source}, {condition}, $\lambda={str(ess)}$")
        # plt.legend(loc='lower right', bbox_to_anchor=(1.3, .05), frameon=False, fontsize=FONTSIZE - 11)
        plt.show()
        plt.savefig(f'{SYS_PATH}/figures/binz2022_gini_vs_trials_{source}_{condition}_exp{experiment_id}_ess{str(ess)}.png')

# def model_ginis_binz2022(pseudo=False, FIGSIZE=(6, 4)):
#     # experiment_id=1
#     # dim = 'dim2' if experiment_id == 3 else 'dim4'
#     # conditions = ['ranked', 'unknown'] if experiment_id ==1 else ['direction', 'unknown'] if experiment_id == 2 else ['unknown']
#     dim = 'dim4'
#     sources = ['claude', 'synthetic']
#     conditions = ['ranked', 'direction', 'unknown']
#     esses = [0.0]
#     ginis = {}
#     for source in sources:
#         for condition in conditions:
#             index = source + '_' + condition
#             for ess in esses:
#                 # ginis[source][condition][ess] = {}
#                 assert ess == 0.0, 'ess must be 0.0'
#                 condition = 'pseudo' + condition if pseudo and source=='claude' and condition != 'unknown' else condition                         
#                 if source == 'claude':
#                     model_coeffs = np.load(f'{PARADIGM_PATH}/data/model_simulation/env={source}_{dim}_{condition}_model=transformer_num_episodes1000000_num_hidden=8_lr0.0003_num_layers=2_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_ess{str(float(ess))}_std0.1_run=0_essinit0.0_annealed_schedulefree_binz2022.npz')['model_coefficients']
#                 elif source == 'synthetic':
#                     model_coeffs = np.load(f'{PARADIGM_PATH}/data/model_simulation/env={source}_{dim}_{condition}_{dim}_model=transformer_num_episodes1000000_num_hidden=8_lr0.0003_num_layers=2_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_ess0.0_std0.1_run=0_{"ranking" if condition == "ranked" else "direction" if condition == "direction" else "unknown"}_essinit0.0_annealed_schedulefree_binz2022.npz')['model_coefficients']
#                 gini_over_tasks = []
#                 for task in range(model_coeffs.shape[1]):
#                     gini_over_tasks.append(gini_compute(np.abs(model_coeffs[:, task, :].mean((0, 1)))))
#                 #ginis[source][condition][ess] = gini_over_tasks
#                 ginis[index] = np.array(gini_over_tasks)
                                        
#     # make a swarm plot of gini coefficients for each condition and source with points being different tasks
#     gini_df = pd.DataFrame.from_dict(ginis)

#     # swarm plot with bar plot for mean gini coefficients for each column
#     f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
#     sns.swarmplot(data=gini_df, ax=ax, color='black', alpha=0.5)
#     sns.barplot(data=gini_df, ax=ax)
#     # color first and third bar with darker shade and second and fourth bar with lighter shade
#     for idx, bar in enumerate(ax.patches):
#         if idx == 0 or idx == 3:
#             bar.set_facecolor('#407193')
#         if idx == 2 or idx == 5:
#             bar.set_facecolor('#527489')
#         elif idx == 1 or idx == 4:
#             bar.set_facecolor('#747875')

#     # Create custom legend handles
#     import matplotlib.patches as mpatches
#     dark_patch = mpatches.Patch(color='#407193', label='Ranking')
#     light_patch = mpatches.Patch(color='#527489', label='Direction')
#     middle_patch = mpatches.Patch(color='#747875', label='Unknown')
#     # Add legend to the plot
#     ax.legend(handles=[dark_patch, light_patch, middle_patch], fontsize=FONTSIZE-2, frameon=False)
#     # Set custom x-ticks and labels
#     ax.set_xticks([1.0, 3.0])
#     ax.set_xticklabels(['ERMI', 'MI'], fontsize=FONTSIZE-2)
#     plt.yticks(fontsize=FONTSIZE-2)
#     # plt.ylabel('Gini Coefficient', fontsize=FONTSIZE)
#     plt.title(f'Gini Coefficient', fontsize=FONTSIZE)
#     sns.despine()
#     plt.savefig(f'{SYS_PATH}/figures/binz2022_gini_coefficients_{dim}_pseudo={pseudo}.png')
#     plt.show()

def model_ginis_binz2022(pseudo=False, FIGSIZE=(10, 4)):
    # experiment_id=1
    # dim = 'dim2' if experiment_id == 3 else 'dim4'
    # conditions = ['ranked', 'unknown'] if experiment_id ==1 else ['direction', 'unknown'] if experiment_id == 2 else ['unknown']
    dim = 'dim4'
    sources = ['claude', 'synthetic']
    conditions = ['ranked', 'direction', 'unknown']
    esses = [0.0]
    ginis = {}
    for source in sources:
        for condition in conditions:
            index = source + '_' + condition
            for ess in esses:
                # ginis[source][condition][ess] = {}
                assert ess == 0.0, 'ess must be 0.0'
                condition = 'pseudo' + condition if pseudo and source=='claude' and condition != 'unknown' else condition                         
                if source == 'claude':
                    model_coeffs = np.load(f'{PARADIGM_PATH}/data/model_simulation/env={source}_{dim}_{condition}_model=transformer_num_episodes1000000_num_hidden=8_lr0.0003_num_layers=2_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_ess{str(float(ess))}_std0.1_run=0_essinit0.0_annealed_schedulefree_binz2022.npz')['model_coefficients']
                elif source == 'synthetic':
                    model_coeffs = np.load(f'{PARADIGM_PATH}/data/model_simulation/env={source}_{dim}_{condition}_{dim}_model=transformer_num_episodes1000000_num_hidden=8_lr0.0003_num_layers=2_d_model=64_num_head=8_noise0.0_shuffleTrue_pairedTrue_lossnll_ess0.0_std0.1_run=0_{"ranking" if condition == "ranked" else "direction" if condition == "direction" else "unknown"}_essinit0.0_annealed_schedulefree_binz2022.npz')['model_coefficients']
                gini_over_tasks = []
                for task in range(model_coeffs.shape[1]):
                    gini_over_tasks.append(gini_compute(np.abs(model_coeffs[:, task, :].mean((0, 1)))))
                #ginis[source][condition][ess] = gini_over_tasks
                ginis[index] = np.array(gini_over_tasks)
                                        
    # make a swarm plot of gini coefficients for each condition and source with points being different tasks
    gini_df = pd.DataFrame.from_dict(ginis)

    # swarm plot with bar plot for mean gini coefficients for each column
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    # sns.swarmplot(data=gini_df, ax=ax, color='black', alpha=0.5)
    sns.barplot(data=gini_df, ax=ax, errorbar='se') 
    
    # Define base colors and patterns
    ermi_base_color = '#407193'  # Blue for ERMI
    mi_base_color = '#CA8243'    # Orange for MI
    
    # Different alpha levels for each condition (ranked=1.0, direction=0.7, unknown=0.4)
    # alphas = [1.0, 1.0, 1.0] #[1.0, 0.7, 0.4]
    alphas = [1.0, 0.7, 0.25]
    
    # Different patterns for each condition
    patterns = ['','','']# ['///', '...', '|||']  # Ranked, Direction, Unknown
    
    # Color and pattern assignment
    for idx, bar in enumerate(ax.patches):
        if idx in [0, 1, 2]:  # ERMI bars (claude)
            condition_idx = idx  # 0=ranked, 1=direction, 2=unknown
            bar.set_facecolor(ermi_base_color)
            bar.set_alpha(alphas[condition_idx])
            bar.set_hatch(patterns[condition_idx])
            bar.set_edgecolor('white')
            bar.set_linewidth(1)
        elif idx in [3, 4, 5]:  # MI bars (synthetic)
            condition_idx = idx - 3  # 0=ranked, 1=direction, 2=unknown
            bar.set_facecolor(mi_base_color)
            bar.set_alpha(alphas[condition_idx])
            bar.set_hatch(patterns[condition_idx])
            bar.set_edgecolor('white')
            bar.set_linewidth(1)

    # Create custom legend handles with alpha and patterns
    import matplotlib.patches as mpatches
    
    # ERMI legend patches
    ermi_ranked_patch = mpatches.Patch(facecolor=ermi_base_color, alpha=alphas[0], 
                                      hatch=patterns[0], edgecolor='white', label='ERMI-Ranked')
    ermi_direction_patch = mpatches.Patch(facecolor=ermi_base_color, alpha=alphas[1], 
                                         hatch=patterns[1], edgecolor='white', label='ERMI-Direction')
    ermi_unknown_patch = mpatches.Patch(facecolor=ermi_base_color, alpha=alphas[2], 
                                       hatch=patterns[2], edgecolor='white', label='ERMI-Unknown')
    
    # MI legend patches
    mi_ranked_patch = mpatches.Patch(facecolor=mi_base_color, alpha=alphas[0], 
                                    hatch=patterns[0], edgecolor='white', label='MI-Ranked')
    mi_direction_patch = mpatches.Patch(facecolor=mi_base_color, alpha=alphas[1], 
                                       hatch=patterns[1], edgecolor='white', label='MI-Direction')
    mi_unknown_patch = mpatches.Patch(facecolor=mi_base_color, alpha=alphas[2], 
                                     hatch=patterns[2], edgecolor='white', label='MI-Unknown')
    
    # Add legend to the plot
    ermi_patch = mpatches.Patch(color='#407193', alpha=1.0, hatch='', edgecolor='white', label='ERMI')
    mi_patch = mpatches.Patch(color='#CA8243', alpha=1.0, hatch='', edgecolor='white', label='MI')
    ax.legend(handles=[ermi_patch, mi_patch], fontsize=FONTSIZE-2, frameon=False, loc='upper center', bbox_to_anchor=(0.1, 1.1), ncol=1)
    # Set custom x-ticks and labels
    # ax.set_xticks([1.0, 4.0])
    ax.set_xticklabels(['Ranking', 'Direction', 'Unknown', 'Ranking', 'Direction', 'Unknown'], fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.ylabel('Gini Coefficient', fontsize=FONTSIZE-2)
    # plt.title(f'Gini Coefficient', fontsize=FONTSIZE)
    sns.despine()
    plt.savefig(f'{SYS_PATH}/figures/binz2022_gini_coefficients_{dim}_pseudo={pseudo}.svg', dpi=300, bbox_inches='tight')
    plt.show()

    
def model_comparison_binz2022(experiment_id, bermi=False, bmi=False, pseudo=False, FIGSIZE= (6,4)):
    
    data = pd.read_csv(f'{PARADIGM_PATH}/data/human/binz2022heuristics_exp{experiment_id}.csv')
    num_participants = data.participant.nunique()
    num_trials = (data.trial.max()+1)*(data.task.max()+1) 

    # load model fit results
    results_mi = np.load(f'{PARADIGM_PATH}/data/model_comparison/task=binz2022_experiment={experiment_id}_source=synthetic_condition=unknown_loss=nll_paired=True_method=unbounded_optimizer=grid_search_numiters=5.npz')
    results_ermi = np.load(f'{PARADIGM_PATH}/data/model_comparison/task=binz2022_experiment={experiment_id}_source=claude_condition=unknown_loss=nll_paired=True_method=unbounded_optimizer=grid_search_numiters=5.npz')
    results_bermi = np.load(f'{PARADIGM_PATH}/data/model_comparison/task=binz2022_experiment={experiment_id}_source=claude_condition=unknown_loss=nll_paired=True_method=bounded_optimizer=grid_search_numiters=5.npz')
    results_bmi = np.load(f'{PARADIGM_PATH}/data/model_comparison/task=binz2022_experiment={experiment_id}_source=synthetic_condition=unknown_loss=nll_paired=True_method=bounded_optimizer=grid_search_numiters=5.npz')
    logprobs_bmi = torch.load(f'{PARADIGM_PATH}/data/model_comparison/logprobs{experiment_id}_bmli.pth', weights_only=False)[0]
    logprobs_baselines = torch.load(f'{PARADIGM_PATH}/data/model_comparison/logprobs{experiment_id}_fitted.pth', weights_only=False)[0]
    logprobs_selection = torch.load(f'{PARADIGM_PATH}/data/model_comparison/logprobs{experiment_id}_selection_fitted.pth', weights_only=False)[0]
    logprobs_feedforward = torch.load(f'{PARADIGM_PATH}/data/model_comparison/logprobs{experiment_id}_feedforward_fitted.pth', weights_only=False)[0]
    logprobs_bmi = torch.cat([logprobs_baselines[:, [0]], logprobs_bmi], dim=-1)
    best_logprobs, best_index = torch.max(logprobs_bmi, dim=-1)
    logprobs_baselines[:, 0] = - 2 * logprobs_baselines[:, 0]
    logprobs_baselines[:, 1:] = - 2 * logprobs_baselines[:, 1:] + 1*np.log(num_trials)

    # compute bic
    bermi_bic = (2*results_bermi['nlls'] + 2*np.log(num_trials))#.sum()
    bmi_bic = (2*results_bmi['nlls'] + 2*np.log(num_trials))#.sum()
    ermi_bic = (2*results_ermi['nlls'] + 1*np.log(num_trials))#.sum()
    mi_bic = (2*results_mi['nlls']+ 1*np.log(num_trials))#.sum() 
    rnn_mi_bic = (-2*logprobs_bmi[:, 1] + 1*np.log(num_trials))#.sum() 
    rnn_bmi_bic = (-2*best_logprobs + 2*np.log(num_trials))#.sum()
    guessing_bic = logprobs_baselines[:, 0]#.sum()
    ideal_bic = logprobs_baselines[:, 1]#.sum()
    equal_bic = logprobs_baselines[:, 2]#.sum()
    single_bic = logprobs_baselines[:, 3]#.sum()
    strategy_bic =  (- 2 * logprobs_selection + 1*np.log(num_trials))#.sum()
    feedforward_bic = (- 2 *  logprobs_feedforward  +  2 * np.log(num_trials))#.sum()
    random_bic_total = -2*np.log(0.5)*num_trials*num_participants
    random_bic_per_participant = -2*np.log(0.5)*num_trials#*num_participants
    
    # experiment specific model fit results
    condition = 'ranked' if experiment_id == 1 else 'direction'
    if experiment_id == 1 or experiment_id == 2:

        results_mi_condition = np.load(f'{PARADIGM_PATH}/data/model_comparison/task=binz2022_experiment={experiment_id}_source=synthetic_condition={condition}_loss=nll_paired=True_method=unbounded_optimizer=grid_search_numiters=5.npz')
        results_bmi_condition = np.load(f'{PARADIGM_PATH}/data/model_comparison/task=binz2022_experiment={experiment_id}_source=synthetic_condition={condition}_loss=nll_paired=True_method=bounded_optimizer=grid_search_numiters=5.npz')
        condition = 'pseudo' + condition if pseudo else condition
        results_ermi_condition = np.load(f'{PARADIGM_PATH}/data/model_comparison/task=binz2022_experiment={experiment_id}_source=claude_condition={condition}_loss=nll_paired=True_method=unbounded_optimizer=grid_search_numiters=5.npz')
        results_bermi_condition = np.load(f'{PARADIGM_PATH}/data/model_comparison/task=binz2022_experiment={experiment_id}_source=claude_condition={condition}_loss=nll_paired=True_method=bounded_optimizer=grid_search_numiters=5.npz')
        
        bermi_condition_bic = (2*results_bermi_condition['nlls'] + 2*np.log(num_trials))#.sum()
        bmi_condition_bic = (2*results_bmi_condition['nlls'] + 2*np.log(num_trials))#.sum()
        ermi_condition_bic = (2*results_ermi_condition['nlls'] + 1*np.log(num_trials))#.sum()
        mi_condition_bic = (2*results_mi_condition['nlls']+ 1*np.log(num_trials))#.sum() 
        abbr = 'R' if 'rank' in condition else 'D'

        # # collect bics and model names
        if bermi:
            bics = [bermi_bic, bmi_bic, ermi_bic, mi_bic, bermi_condition_bic, bmi_condition_bic, ermi_condition_bic, mi_condition_bic]
            models = ['BERMI', 'BMI', 'ERMI', 'MI', f'BERMI-{abbr}', f'BMI-{abbr}', f'ERMI-{abbr}', f'MI-{abbr}']
        elif bmi:
            # collect bics and model names
            bics = [bmi_condition_bic, ermi_condition_bic, mi_condition_bic]
            models = [f'BMI', f'ERMI', f'MI']
        else:
            # collect bics and model names
            bics = [ermi_condition_bic, mi_condition_bic]
            models = [f'ERMI', f'MI']

    else:   
        
        if bermi:
            # collect bics and model names
            bics = [bermi_bic, bmi_bic, ermi_bic, mi_bic]#, rnn_mi_bic, rnn_bmi_bic, random_bic]
            models = ['BERMI', 'BMI', 'ERMI', 'MI']#,  'RNN_MI', 'RNN_BMI', 'random']
        elif bmi:
            # collect bics and model names
            # bics = [bmi_bic, ermi_bic, mi_bic]
            # models = ['BMI', 'ERMI', 'MI']
            bics = [ermi_bic, bmi_bic, mi_bic, equal_bic, single_bic, feedforward_bic] #ideal_bic,  strategy_bic
            models = [f'ERMI', f'BMI', f'MI', 'Equal Weighting', 'Single Cue', 'Feedforward Network']#'Ideal Observer',  'Strategy Selection',
        else:
            # collect bics and model names
            # bics = [bmi_bic, ermi_bic, mi_bic]
            # models = ['BMI', 'ERMI', 'MI']
            bics = [ermi_bic, mi_bic, equal_bic, single_bic, feedforward_bic]#, ideal_bic,  strategy_bic]
            # models = [f'ERMI', f'MI', 'Equal Weighting', 'Single Cue', 'Feedforward Network']#'Ideal Observer',  'Strategy Selection',
            models = [f'ERMI', f'MI', 'EW', 'SC', 'NN']#, 'IO',  'SS']
    
    # apply .sum to all bics
    total_bics = [bic.sum() for bic in bics]
    # sort bics and models
    colors = ['#407193', '#CA8243','#505050','#505050', '#505050', '#505050', '#505050', '#505050', '#505050'][:len(bics)]
    total_bics, models, colors = zip(*sorted(zip(total_bics, models, colors)))
    # colors = ['#173b4f', '#8b9da7', '#173b4f', '#8b9da7', '#5d7684', '#2f4a5a', "#161717", '#4d6a75', '#748995', '#a2c0a9', '#c4d9c2'][:len(bics)]
    # colors = ['#173b4f', '#8b9da7', '#173b4f', '#8b9da7', '#5d7684', '#2f4a5a', '#0d2c3d', '#4d6a75', '#748995', '#a2c0a9', '#c4d9c2'][:len(bics)]
    # compare mean BICS across models in a bar plot
    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    bar_positions = np.arange(len(total_bics))*1.5
    ax.bar(bar_positions, np.array(total_bics), color=colors, width=1.)
    ax.set_xlabel('Models', fontsize=FONTSIZE)
    ax.set_ylabel('BIC', fontsize=FONTSIZE)
    ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
    ax.set_xticklabels(models, fontsize=FONTSIZE-6)  # Assign category names to x-tick labels
    # ax.set_title(f'experiment {experiment_id}', fontsize=FONTSIZE)
    # ax.set_title(f'{"Ranking" if experiment_id == 1 else "Direction" if experiment_id == 2 else "Unknown (Dim=2)" if experiment_id == 3 else "Unknown (Dim=4)"}', fontsize=FONTSIZE)
    ax.axhline(y=random_bic_total, color='red', linestyle='dotted', lw=3)
    plt.yticks(fontsize=FONTSIZE-2)
    plt.ylim(min(total_bics)-100, random_bic_total+100)
    sns.despine()
    f.tight_layout()
    plt.savefig(f'{SYS_PATH}/figures/binz2022_model_comparison_exp{experiment_id}.svg')
    plt.show()

    # posteior model frequency
    posterior_model_frequency(np.array(bics), models, colors, horizontal=False, FIGSIZE=(6,4), task_name=f'Binz2022_exp{experiment_id}')
    # exceedance probability
    exceedance_probability(np.array(bics), models, colors, horizontal=False, FIGSIZE=(6,4), task_name=f'Binz2022_exp{experiment_id}')
    

def posterior_model_frequency(bics, models, colors, horizontal=False, FIGSIZE=(5,4), task_name=None):
    result = {}
    LogEvidence = np.stack(-bics/2)
    result = GroupBMC(LogEvidence).get_result()
    # rename models for plot
    # colors = ['#173b4f', '#4d6a75','#5d7684', '#748995','#4d6a75', '#0d2c3d', '#a2c0a9', '#2f4a5a', '#8b9da7', '#c4d9c2'][:bics.shape[0]]
    # colors = ['#407193', '#CA8243','#505050','#505050', '#505050', '#505050', '#505050', '#505050', '#505050'][:bics.shape[0]]
    # sort result in descending order
    sort_order = np.argsort(result.frequency_mean)[::-1]
    result.frequency_mean = result.frequency_mean[sort_order]
    result.frequency_var = result.frequency_var[sort_order]
    models = np.array(models)[sort_order]
    colors = np.array(colors)[sort_order]

    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    
    if horizontal:
        # composed
        ax.barh(np.arange(len(models)), result.frequency_mean, xerr=np.sqrt(result.frequency_var), align='center', color=colors, height=0.6)#, edgecolor='k')#, hatch='//', label='Compostional Subtask')
        # plt.legend(fontsize=FONTSIZE-4, frameon=False)
        ax.set_ylabel('Models', fontsize=FONTSIZE)
        # ax.set_xlim(0, 0.7)
        ax.set_xlabel('Posterior model frequency', fontsize=FONTSIZE) 
        plt.yticks(ticks=np.arange(len(models)), labels=models, fontsize=FONTSIZE-2)
        ax.set_xticks(np.arange(0, result.frequency_mean.max(), 0.1))
        plt.xticks(fontsize=FONTSIZE-2)
    else:
        # bar_positions = np.arange(len(result.frequency_mean))*0.5
        # ax.bar(bar_positions, result.frequency_mean, color=colors, width=0.4)
        # ax.errorbar(bar_positions, result.frequency_mean, yerr= np.sqrt(result.frequency_var), c='k', lw=3, fmt="o")
        # ax.set_xlabel('Models', fontsize=FONTSIZE)
        # ax.set_ylabel('Model frequency', fontsize=FONTSIZE)
        # ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
        # ax.set_xticklabels(models, fontsize=FONTSIZE-2)  # Assign category names to x-tick labels
        # plt.yticks(fontsize=FONTSIZE-2)
        # # start bar plot from 0
        # ax.set_ylim([-0.01, .9])
        # # y ticks at 0.1 interval
        # ax.set_yticks(np.arange(0.0, .90, 0.2))
        # Create a DataFrame for seaborn
        # colors = assign_colors_by_model(models)
        df_freq = pd.DataFrame({
            'Models': models,
            'frequency_mean': result.frequency_mean,
            'frequency_var': result.frequency_var,
            'colors': colors
        })

        # Create the barplot with error bars
        sns.barplot(data=df_freq, x='Models', y='frequency_mean', palette=colors, ax=ax, errorbar=None)

        # Add custom error bars
        ax.errorbar(range(len(models)), result.frequency_mean, yerr=np.sqrt(result.frequency_var), 
                c='k', lw=3, fmt="o", capsize=5)

        ax.set_xlabel('', fontsize=FONTSIZE-2)
        ax.set_ylabel('Model frequency', fontsize=FONTSIZE-2)
        plt.xticks(fontsize=FONTSIZE-2)
        plt.yticks(fontsize=FONTSIZE-2)
        # start bar plot from 0
        ax.set_ylim([-0.01, .9])
        # y ticks at 0.1 interval
        ax.set_yticks(np.arange(0.0, .90, 0.2))

    # ax.set_title(f'Model Comparison', fontsize=FONTSIZE)
    # print model names, mean frequencies and std error of mean frequencies
    for i, model in enumerate(models):
        print(f'{model}: {result.frequency_mean[i]} +- {np.sqrt(result.frequency_var[i])}')

    sns.despine()
    f.tight_layout()
    f.savefig(f'{SYS_PATH}/figures/posterior_model_frequency_{task_name}.svg', bbox_inches='tight', dpi=300)
    plt.show()

def exceedance_probability(bics, models, colors, horizontal=False, FIGSIZE=(7,5), task_name=None):
    result = {}
    LogEvidence = np.stack(-bics/2)
    result = GroupBMC(LogEvidence).get_result()

    # rename models for plot
    # colors = ['#173b4f', '#8b9da7', '#5d7684', '#2f4a5a', '#0d2c3d', '#4d6a75', '#748995', '#a2c0a9', '#c4d9c2', '#3b3b3b', '#c4d9c2'][:bics.shape[0]]
    # sort result in descending order
    sort_order = np.argsort(result.exceedance_probability)[::-1]
    result.exceedance_probability = result.exceedance_probability[sort_order]
    models = np.array(models)[sort_order]
    colors = np.array(colors)[sort_order]

    f, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    if horizontal:
        # composed
        ax.barh(np.arange(len(models)), result.exceedance_probability, align='center', color=colors[:len(models)], height=0.6)#, hatch='//', label='Compostional Subtask')
        # plt.legend(fontsize=FONTSIZE-4, frameon=False)
        ax.set_ylabel('Models', fontsize=FONTSIZE)
        # ax.set_xlim(0, 0.7)
        ax.set_xlabel('Exceedance probability', fontsize=FONTSIZE) 
        plt.yticks(ticks=np.arange(len(models)), labels=models, fontsize=FONTSIZE-3.)
        # ax.set_xticks(np.arange(0, result.exceedance_probability.max(), 0.1))
        plt.xticks(fontsize=FONTSIZE-4)
    else:
        # composed
        bar_positions = np.arange(len(result.exceedance_probability))*0.5
        ax.bar(bar_positions, result.exceedance_probability, color=colors, width=0.4)
        # plt.legend(fontsize=FONTSIZE, frameon=False)
        # ax.set_xlabel('Models', fontsize=FONTSIZE)
        # ax.set_ylim(0, 0.7)
        ax.set_ylabel('Exceedance probability', fontsize=FONTSIZE) 
        ax.set_xticks(bar_positions)  # Set x-tick positions to bar_positions
        ax.set_xticklabels(models, fontsize=FONTSIZE-2)  # Assign category names to x-tick labels
        plt.yticks(fontsize=FONTSIZE-2)
    
    # ax.set_title(f'Model Comparison', fontsize=FONTSIZE)
    sns.despine()
    f.tight_layout()
    f.savefig(f'{SYS_PATH}/figures/exceedance_probability_{task_name}.svg', bbox_inches='tight', dpi=300)
    plt.show()