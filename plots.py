import json
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import math
import json
from scipy.stats import ttest_ind
from utils import get_stat_func, CompressPDF

font = {'size': 80}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.constrained_layout.use'] = True

colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098],  [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
          [0.494, 0.1844, 0.556],[0.3010, 0.745, 0.933], [137/255,145/255,145/255],
          [0.466, 0.674, 0.8], [0.929, 0.04, 0.125],
          [0.3010, 0.245, 0.33], [0.635, 0.078, 0.184], [0.35, 0.78, 0.504]]

# [[0, 0.447, 0.7410], [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],  # c2[0.85, 0.325, 0.098],[0.85, 0.325, 0.098],
#  [0.494, 0.1844, 0.556], [209 / 255, 70 / 255, 70 / 255], [137 / 255, 145 / 255, 145 / 255],  # [0.3010, 0.745, 0.933],
#  [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
#  [0.3010, 0.745, 0.933], [0.635, 0.078, 0.184]]

RESULTS_PATH = '/home/ahmed/Documents/final_year/ALOE2022/rlgraph/rebuttal_results_semantic/'
SAVE_PATH = '/home/ahmed/Documents/final_year/ALOE2022/rlgraph/plots/'
TO_PLOT = ['semantic_goals']

NB_CLASSES = 11 # 12 for 5 blocks

LINE = 'mean'
ERR = 'std'
DPI = 30
N_SEEDS = None
N_EPOCHS = None
LINEWIDTH = 8 # 8 for per class
MARKERSIZE = 15 # 15 for per class
ALPHA = 0.3
ALPHA_TEST = 0.05
MARKERS = ['o', 'v', 's', 'P', 'D', 'X', "*", 'v', 's', 'p', 'P', '1']
FREQ = 5
NB_BUCKETS = 10
NB_EPS_PER_EPOCH = 2400
NB_VALID_GOALS = 35
LAST_EP = 165
LIM = NB_EPS_PER_EPOCH * LAST_EP / 1000 + 5
line, err_min, err_plus = get_stat_func(line=LINE, err=ERR)
COMPRESSOR = CompressPDF(4)
# 0: '/default',
# 1: '/prepress',
# 2: '/printer',
# 3: '/ebook',
# 4: '/screen'


def setup_figure(xlabel=None, ylabel=None, xlim=None, ylim=None):
    fig = plt.figure(figsize=(30, 20), frameon=False) # 34 18 for semantic
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=10, direction='in', length=20, labelsize='40')
    artists = ()
    if xlabel:
        xlab = plt.xlabel(xlabel)
        artists += (xlab,)
    if ylabel:
        ylab = plt.ylabel(ylabel)
        artists += (ylab,)
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    return artists, ax

def setup_n_figs(n, m, xlabels=None, ylabels=None, xlims=None, ylims=None):
    fig, axs = plt.subplots(n, m, figsize=(48, 12), frameon=False)
    axs = axs.ravel()
    artists = ()
    for i_ax, ax in enumerate(axs):
        ax.spines['top'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.tick_params(width=7, direction='in', length=15, labelsize='55', zorder=10)
        if xlabels[i_ax]:
            xlab = ax.set_xlabel(xlabels[i_ax])
            artists += (xlab,)
        if ylabels[i_ax]:
            ylab = ax.set_ylabel(ylabels[i_ax])
            artists += (ylab,)
        if ylims[i_ax]:
            ax.set_ylim(ylims[i_ax])
        if xlims[i_ax]:
            ax.set_xlim(xlims[i_ax])
    return fig, artists, axs

def save_fig(path, artists):
    plt.savefig(os.path.join(path), bbox_extra_artists=artists, bbox_inches='tight', dpi=DPI)
    plt.close('all')
    # compress PDF
    try:
        COMPRESSOR.compress(path, path[:-4] + '_compressed.pdf')
        os.remove(path)
    except:
        pass


def check_length_and_seeds(experiment_path):
    conditions = os.listdir(experiment_path)
    # check max_length and nb seeds
    max_len = 0
    max_seeds = 0
    min_len = 1e6
    min_seeds = 1e6

    for cond in conditions:
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        if len(list_runs) > max_seeds:
            max_seeds = len(list_runs)
        if len(list_runs) < min_seeds:
            min_seeds = len(list_runs)
        for run in list_runs:
            try:
                run_path = cond_path + run + '/'
                data_run = pd.read_csv(run_path + 'progress.csv')
                nb_epochs = len(data_run)
                if nb_epochs > max_len:
                    max_len = nb_epochs
                if nb_epochs < min_len:
                    min_len = nb_epochs
            except:
                pass
    return max_len, max_seeds, min_len, min_seeds

def plot_sr_av(max_len, experiment_path, folder):

    condition_path = experiment_path + folder + '/'
    list_runs = sorted(os.listdir(condition_path))
    global_sr = np.zeros([len(list_runs), max_len])
    global_sr.fill(np.nan)
    sr_data = np.zeros([len(list_runs), NB_CLASSES, max_len])
    sr_data.fill(np.nan)
    x_eps = np.arange(0, (LAST_EP + 1) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
    # x_eps = np.arange(0, max_len, FREQ)
    x = np.arange(0, LAST_EP + 1, FREQ)
    for i_run, run in enumerate(list_runs):
        run_path = condition_path + run + '/'
        data_run = pd.read_csv(run_path + 'progress.csv')

        T = len(data_run['Eval_SR_1'][:LAST_EP + 1])
        SR = np.zeros([NB_CLASSES, T])
        for t in range(T):
            for i in range(NB_CLASSES):
                SR[i, t] = data_run['Eval_SR_{}'.format(i+1)][t]
        all_sr = np.mean([data_run['Eval_SR_{}'.format(i+1)] for i in range(NB_CLASSES)], axis=0)

        sr_buckets = []
        for i in range(SR.shape[0]):
            sr_buckets.append(SR[i])
        sr_buckets = np.array(sr_buckets)
        sr_data[i_run, :, :sr_buckets.shape[1]] = sr_buckets.copy()
        global_sr[i_run, :all_sr.size] = all_sr.copy()

    artists, ax = setup_figure(  # xlabel='Episodes (x$10^3$)',
        xlabel='Episodes (x$10^3$)',
        ylabel='Success Rate',
        xlim=[-1, LIM],
        ylim=[-0.02, 1.03])
    sr_per_cond_stats = np.zeros([NB_CLASSES, max_len, 3])
    sr_per_cond_stats[:, :, 0] = line(sr_data)
    sr_per_cond_stats[:, :, 1] = err_min(sr_data)
    sr_per_cond_stats[:, :, 2] = err_plus(sr_data)
    av = line(global_sr)
    for i in range(NB_CLASSES):
        plt.plot(x_eps, sr_per_cond_stats[i, x, 0], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)
    plt.plot(x_eps, av[x], color=[0.3]*3, linestyle='--', linewidth=LINEWIDTH // 2)
    leg = plt.legend(['Class {}'.format(i+1) for i in range(NB_CLASSES)] + ['Global'],
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.15),
                     ncol=6,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 30, 'weight': 'bold'},
                     markerscale=1)
    artists += (leg,)

    for i in range(NB_CLASSES):
        plt.fill_between(x_eps, sr_per_cond_stats[i, x, 1], sr_per_cond_stats[i, x, 2], color=colors[i], alpha=ALPHA)

    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    plt.grid()
    # ax.set_facecolor((244/255, 244/255, 244/255))
    save_fig(path=SAVE_PATH + folder + '_sr.pdf', artists=artists)


def plot_sr_av_all(max_len, experiment_path):
    fig, artists, ax = setup_n_figs(n=1,
                                   m=3, 
                                #    xlabels=[None, None, 'Episodes (x$10^3$)', 'Episodes (x$10^3$)'],
                                #    ylabels= ['Success Rate', None] * 2,
                                   xlabels = ['Episodes (x$10^3$)'] * 3,
                                   ylabels = ['Success Rate', None, None],
                                   xlims = [[-1, LIM] for _ in range(4)],
                                   ylims= [[-0.02, 1.03] for _ in range(4)]
        )
    # titles = ['Continuous-GN', 'Continuous-IN', 'Continuous-RN', 'Continuous-DS', 'Continuous-Flat']
    titles = ['Continuous-GN', 'Continuous-IN', 'Continuous-DS']
    for k, folder in enumerate(['continuous_full_gn', 'continuous_interaction_network_2', 'continuous_deep_sets']):
        condition_path = experiment_path + folder + '/'
        list_runs = sorted(os.listdir(condition_path))
        global_sr = np.zeros([len(list_runs), max_len])
        global_sr.fill(np.nan)
        sr_data = np.zeros([len(list_runs), NB_CLASSES, max_len])
        sr_data.fill(np.nan)
        x_eps = np.arange(0, (LAST_EP + 1) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
        # x_eps = np.arange(0, max_len, FREQ)
        x = np.arange(0, LAST_EP + 1, FREQ)
        for i_run, run in enumerate(list_runs):
            run_path = condition_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')

            T = len(data_run['Eval_SR_1'][:LAST_EP + 1])
            SR = np.zeros([NB_CLASSES, T])
            for t in range(T):
                for i in range(NB_CLASSES):
                    SR[i, t] = data_run['Eval_SR_{}'.format(i+1)][t]
            all_sr = np.mean([data_run['Eval_SR_{}'.format(i+1)] for i in range(NB_CLASSES)], axis=0)

            sr_buckets = []
            for i in range(SR.shape[0]):
                sr_buckets.append(SR[i])
            sr_buckets = np.array(sr_buckets)
            sr_data[i_run, :, :sr_buckets.shape[1]] = sr_buckets.copy()
            global_sr[i_run, :all_sr.size] = all_sr.copy()
        
        sr_per_cond_stats = np.zeros([NB_CLASSES, max_len, 3])
        sr_per_cond_stats[:, :, 0] = line(sr_data)
        sr_per_cond_stats[:, :, 1] = err_min(sr_data)
        sr_per_cond_stats[:, :, 2] = err_plus(sr_data)
        av = line(global_sr)
        for i in range(NB_CLASSES):
            ax[k].plot(x_eps, sr_per_cond_stats[i, x, 0], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)
        ax[k].plot(x_eps, av[x], color=[0.3]*3, linestyle='--', linewidth=LINEWIDTH // 2)

        for i in range(NB_CLASSES):
            ax[k].fill_between(x_eps, sr_per_cond_stats[i, x, 1], sr_per_cond_stats[i, x, 2], color=colors[i], alpha=ALPHA)

        ax[k].set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax[k].grid()
        ax[k].set_title(titles[k], fontname='monospace', fontweight='bold')
    # ax.set_facecolor((244/255, 244/255, 244/255))
    leg = fig.legend(#['$C_1$', '$C_2$', '$C_3$', '$S_2$', '$S_3$', '$S_2$ & $S_2$', '$S_2$ & $S_3$', '$P_3$', '$P_3$ & $S_2$', '$S_4$', '$S_5$', 'Global'],
                    ['No Stacks', '$\widetilde{S}_2$', '$\widetilde{S}_3$', '$\widetilde{S}_4$', '$\widetilde{S}_5$', 'Global'],
                    loc='upper center',
                    bbox_to_anchor=(0.525, 1.22),
                    ncol=6,
                    fancybox=True,
                    shadow=True,
                    prop={'size': 65, 'weight': 'normal'},
                    markerscale=1)
    artists += (leg,)
    save_fig(path=SAVE_PATH + 'per_class.pdf', artists=artists)


def get_mean_sr(experiment_path, max_len, max_seeds, conditions=None, labels=None, ref='with_init'):
    if conditions is None:
        conditions = os.listdir(experiment_path)
    sr = np.zeros([max_seeds, len(conditions), LAST_EP + 1 ])
    sr.fill(np.nan)
    for i_cond, cond in enumerate(conditions):
        if cond == ref:
            ref_id = i_cond
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        for i_run, run in enumerate(list_runs):
            run_path = cond_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            all_sr = np.mean(np.array([data_run['Eval_SR_{}'.format(i+1)][:LAST_EP + 1] for i in range(NB_CLASSES)]), axis=0)
            sr[i_run, i_cond, :all_sr.size] = all_sr.copy()


    sr_per_cond_stats = np.zeros([len(conditions), LAST_EP + 1, 3])
    sr_per_cond_stats[:, :, 0] = line(sr)
    sr_per_cond_stats[:, :, 1] = err_min(sr)
    sr_per_cond_stats[:, :, 2] = err_plus(sr)


    x_eps = np.arange(0, (LAST_EP + 1) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
    x = np.arange(0, LAST_EP + 1, FREQ)
    artists, ax = setup_figure(xlabel='Episodes (x$10^3$)',
                               # xlabel='Epochs',
                               ylabel='Success Rate',
                               xlim=[-1, LIM],
                               ylim=[-0.02, 1 -0.02 + 0.04 * (len(conditions) + 1)])

    for i in range(len(conditions)):
        plt.plot(x_eps, sr_per_cond_stats[i, x, 0], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)

    if labels is None:
        labels = conditions
    leg = plt.legend(labels,
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.15),
                     ncol=5,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 50, 'weight': 'bold'},
                     markerscale=1,
                     )
    for l in leg.get_lines():
        l.set_linewidth(7.0)
    artists += (leg,)
    for i in range(len(conditions)):
        plt.fill_between(x_eps, sr_per_cond_stats[i, x, 1], sr_per_cond_stats[i, x, 2], color=colors[i], alpha=ALPHA)
    
    # compute p value wrt ref id
    p_vals = dict()
    for i_cond in range(len(conditions)):
        if i_cond != ref_id:
            p_vals[i_cond] = []
            for i in x:
                ref_inds = np.argwhere(~np.isnan(sr[:, ref_id, i])).flatten()
                other_inds = np.argwhere(~np.isnan(sr[:, i_cond, i])).flatten()
                if ref_inds.size > 1 and other_inds.size > 1:
                    ref = sr[:, ref_id, i][ref_inds]
                    other = sr[:, i_cond, i][other_inds]
                    p_vals[i_cond].append(ttest_ind(ref, other, equal_var=False)[1])
                else:
                    p_vals[i_cond].append(1)
                    
    for i_cond in range(len(conditions)):
        if i_cond != ref_id:
            inds_sign = np.argwhere(np.array(p_vals[i_cond]) < ALPHA_TEST).flatten()
            if inds_sign.size > 0:
                plt.scatter(x=x_eps[inds_sign], y=np.ones([inds_sign.size]) - 0.04 + 0.05 * i_cond, marker='*', color=colors[i_cond], s=1300)

    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    plt.grid()
    # ax.set_facecolor((244/255, 244/255, 244/255))
    save_fig(path=SAVE_PATH + PLOT + '.pdf', artists=artists)
    return sr_per_cond_stats.copy()

if __name__ == '__main__':

    for PLOT in TO_PLOT:
        print('\n\tPlotting', PLOT)
        experiment_path = RESULTS_PATH

        max_len, max_seeds, min_len, min_seeds = check_length_and_seeds(experiment_path=experiment_path)

        conditions = [f'{s}' for s in ['full_gn', 'interaction_network_2', 'relation_network', 'deep_sets', 'flat']]
        labels = [f'S-{s}' for s in ['GN', 'IN', 'RN', 'DS', 'Flat']]
        get_mean_sr(experiment_path, max_len, max_seeds, conditions, labels, ref='full_gn')
        # plot_sr_av(max_len, experiment_path, 'flat')
        # plot_sr_av_all(max_len, experiment_path)
