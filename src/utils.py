import ot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import pandas as pd
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from sklearn import preprocessing
import matplotlib.colors as clr
from scipy.stats import vonmises_fisher

def calc_sus(u, v, reg_m=100, weight="uniform", div="l2"):
    if weight == "uniform":
        a = np.ones(len(u)) / len(u)
        b = np.ones(len(v)) / len(v)
    elif weight == "norm":
        a = np.linalg.norm(u, axis=1)
        a /= np.sum(a)
        b = np.linalg.norm(v, axis=1) 
        b /= np.sum(b)
    else:
        raise ValueError("Invalid weight option. Use 'uniform' or 'norm'.")
    u_normalized = u / np.linalg.norm(u, axis=1, keepdims=True)
    v_normalized = v / np.linalg.norm(v, axis=1, keepdims=True)
    C = 1 - np.dot(u_normalized, v_normalized.T)
    T = ot.unbalanced.mm_unbalanced(a, b, C, reg_m=reg_m, div=div)

    u_sus = - (a - T.sum(axis=1)) / a
    v_sus = (b - T.sum(axis=0)) / b
    return u_sus, v_sus

def calc_ldr(u, v):
    u_normalized = u / np.linalg.norm(u, axis=1, keepdims=True)
    v_normalized = v / np.linalg.norm(v, axis=1, keepdims=True)
    mu_s = u_normalized.mean(axis=0)
    mu_t = v_normalized.mean(axis=0)
    ls = np.linalg.norm(mu_s)
    lt = np.linalg.norm(mu_t)
    mu_s  = mu_s / ls
    mu_t = mu_t / lt
    ks = ls * (len(mu_s)-ls**2) / (1-ls**2)
    kt = lt * (len(mu_s)-lt**2) / (1-lt**2)
    s_LDR = vonmises_fisher(mu_t, kt).logpdf(u_normalized) - vonmises_fisher(mu_s, ks).logpdf(u_normalized)
    t_LDR = vonmises_fisher(mu_t, kt).logpdf(v_normalized) - vonmises_fisher(mu_s, ks).logpdf(v_normalized)
    return s_LDR, t_LDR

def calc_OT_and_UOT_matrix(u, v, reg_m=100, weight="uniform", div="l2"):
    u_normalized = u / np.linalg.norm(u, axis=1, keepdims=True)
    v_normalized = v / np.linalg.norm(v, axis=1, keepdims=True)
    C = 1 - np.dot(u_normalized, v_normalized.T)
    a = np.ones(len(u)) / len(u)
    b = np.ones(len(v)) / len(v)
    T_ot = ot.emd(a, b, C)
    T_uot = ot.unbalanced.mm_unbalanced(a, b, C, reg_m=reg_m, div=div)
    return T_ot, T_uot


def pallet2cmap(my_pallet):
    color_index = my_pallet[:, 0].astype(np.float32)
    color_index = preprocessing.minmax_scale(color_index)
    color_index[0] = 0
    color_index[-1] = 1 # to avoid floating point error and scale to 0-1
    color_norm = []
    for idx, cdx in enumerate(color_index):
        color_norm.append([cdx, my_pallet[idx, 1]])
    cmap = clr.LinearSegmentedColormap.from_list('color', color_norm, N=256)
    return cmap

def create_concat_one_cmap(sus_list):
    '''
    input:
    sus list は複数単語の sus のリスト
    '''
    threshold_low = min([np.quantile(np.abs(sus), 0.3) for sus in sus_list])
    threshold_high = max([np.quantile(np.abs(sus), 0.95) for sus in sus_list])
    max_abs_sus = max([np.max(np.abs(sus)) for sus in sus_list])
    color = np.array([
        [-max_abs_sus, '#0000ff'],
        [-threshold_high, '#0000ff'],
        [-threshold_low, '#ffffff'],
        [threshold_low, '#ffffff'],
        [threshold_high, '#ff0000'],
        [max_abs_sus, '#ff0000']
    ])
    cmap = pallet2cmap(color)
    return cmap

def custom_formatter(x, pos, log=True):
    if log:
        if x > 0:
            return r'$+10^{:.0f}$'.format(x)
        elif x < 0:
            return r'$-10^{:.0f}$'.format(np.abs(x))
        else:
            return r'$\pm 0$'
    else:
        if x > 0:
            return r'$+{:.1f}$'.format(x)
        elif x < 0:
            return r'${:.1f}$'.format(x)
        else:
            return r'$\pm0.0$'

def scatter_plot(u, v, u_sus, v_sus, fig, ax, legend=True, max_abs_sus=None, cmap=None, ldr=False):
    all_sus = np.hstack([u_sus, v_sus])
    all_vecs = np.vstack([u, v])
    if max_abs_sus is None:
        max_abs_sus = max(np.abs(all_sus))
    

    divider = make_axes_locatable(ax)
    if cmap is None:
        cmap = create_concat_one_cmap([all_sus])

    all_sus = np.append(all_sus, [max_abs_sus, -max_abs_sus])
    all_vecs = np.vstack([all_vecs, np.array([[0,0], [0,0]])])
    p = ax.scatter(all_vecs[:, 0], all_vecs[:, 1], c=all_sus, cmap=cmap, s=0)
    
    cax = divider.append_axes('right', size='3%', pad=0.1)
    cbb = fig.colorbar(p, cax)
    cbb.set_label('Sense Usage Shift', fontsize=20, labelpad=25, loc='center', rotation=270)
    cbb.ax.tick_params(labelsize=15)
    if ldr:
        cbar_locator = MultipleLocator(1.0)  
        cbb.ax.yaxis.set_major_locator(cbar_locator)
        cbb.ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    else:
        cbar_locator = MultipleLocator(0.2)  
        cbb.ax.yaxis.set_major_locator(cbar_locator)


    all_sus = all_sus[:-2]
    all_vecs = all_vecs[:-2]
    norm = Normalize(vmin=-max_abs_sus, vmax=max_abs_sus)
    colors = cmap(norm(all_sus))
    df = pd.DataFrame(all_vecs, columns=['x', 'y'])
    df['score'] = np.append(-u_sus, v_sus)
    
    df['marker'] = ['o']*len(u) + ['s']*len(v)
    df['color'] = colors.tolist()
    df = df.sort_values(by='score', key=lambda x: abs(x), ascending=True)

    
    for i, row in df.iterrows():
        if row['score'] > 0:
            ax.scatter(row['x'], row['y'], c=row['color'], edgecolors="black", linewidths=0.5, marker=row['marker'], s=90)
        else:
            ax.scatter(row['x'], row['y'], c=row['color'],
                   edgecolors="black", linewidths=0.5, marker=row['marker'], s=90)

    if legend:
        ax.scatter([], [], s=90, c="blue", edgecolors="black", linewidths=0.5, label=r"1810―1860")
        ax.scatter([], [], s=90, c="red", edgecolors="black", linewidths=0.5, marker="s", label=r"1960―2010")
        ax.legend(fontsize=15)
    
    ax.set_aspect('equal')
    return cax

def multiple_scatter_plot(u_list, v_list, u_sus_list, v_sus_list, fig, axs, tgt_words, legend=True):
    all_sus = np.hstack([np.hstack(u_sus_list), np.hstack(v_sus_list)])
    all_vecs = np.vstack([np.vstack(u_list), np.vstack(v_list)])
    min_x = min(all_vecs[:, 0])-1
    max_x = max(all_vecs[:, 0])+1
    min_y = min(all_vecs[:, 1])-1
    max_y = max(all_vecs[:, 1])+1
    max_abs_sus = max(np.abs(all_sus))
    cmap = create_concat_one_cmap([all_sus])

    for i in range(len(u_list)):
        u = u_list[i]
        v = v_list[i]
        u_sus = u_sus_list[i]
        v_sus = v_sus_list[i]
        ax = axs[i]
        scatter_plot(u, v, u_sus, v_sus, fig, ax, legend=legend, max_abs_sus=max_abs_sus, cmap=cmap)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_title(tgt_words[i], fontsize=20)