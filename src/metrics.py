import numpy as np
import os, sys
import ot

from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from collections import Counter
from scipy.spatial.distance import cdist, cosine

import utils

working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(working_dir)
from CSSDetection.src.app import APosterioriaffinityPropagation


def jsd(L1, L2):
    labels = np.unique(np.concatenate([L1, L2]))
    c1 = Counter(L1)
    c2 = Counter(L2)
    L1_dist = np.array([c1[l] for l in labels])
    L2_dist = np.array([c2[l] for l in labels])
    L1_dist = L1_dist / L1_dist.sum()
    L2_dist = L2_dist / L2_dist.sum()
    return jensenshannon(L1_dist, L2_dist) 


def entropy_diff(L1, L2):
    labels = np.unique(np.concatenate([L1, L2]))
    c1 = Counter(L1)
    c2 = Counter(L2)
    L1_dist = np.array([c1[l] for l in labels])
    L2_dist = np.array([c2[l] for l in labels])
    L1_dist = L1_dist / L1_dist.sum()
    L2_dist = L2_dist / L2_dist.sum()
    return entropy(L2_dist) - entropy(L1_dist)

def apdp(u, v, L1, L2):
    # cluster centroids
    u = np.array(u)
    v = np.array(v)
    mu_E1 = np.array([u[L1 == label].mean(axis=0) for label in np.unique(L1)])
    mu_E2 = np.array([v[L2 == label].mean(axis=0) for label in np.unique(L2)])
    return np.mean(cdist(mu_E1, mu_E2, metric="canberra"))

def widid(u, v, damping=0.5):
    # return two lists of labels
    app = APosterioriaffinityPropagation(affinity='cosine', 
                           damping=damping,
                           max_iter=200,
                           convergence_iter=15,
                           copy=True,
                           preference=None,
                           random_state=42)
    app.fit(np.array(u))
    app.fit(np.array(v))
    return app.labels_[:len(u)], app.labels_[len(u):]


def f_SUS(u, v, reg_m=100):
    s_sus, t_sus = utils.calc_sus(u, v, reg_m=reg_m)
    return np.abs(np.mean(s_sus) - np.mean(t_sus))

def f_1(u, v, reg_m=100):
    s_sus, t_sus = utils.calc_sus(u, v, reg_m=reg_m)
    return np.abs(s_sus).sum() + np.abs(t_sus).sum()

def f_2(u, v, reg_m=100, theta=0.5):
    s_sus, t_sus = utils.calc_sus(u, v, reg_m=reg_m)
    threshold = max(max(np.abs(s_sus)), max(np.abs(t_sus))) * theta
    return - s_sus[s_sus < -threshold].sum() + t_sus[t_sus > threshold].sum()

def f_3(u, v, reg_m=100):
    u_normalized = u / np.linalg.norm(u, axis=1, keepdims=True)
    v_normalized = v / np.linalg.norm(v, axis=1, keepdims=True)
    C = 1 - np.dot(u_normalized, v_normalized.T)
    a = np.ones(len(u)) / len(u)
    b = np.ones(len(v)) / len(v)
    return ot.unbalanced.mm_unbalanced2(a, b, C, reg_m=reg_m, div="l2")

def f_OT(u, v):
    u_normalized = u / np.linalg.norm(u, axis=1, keepdims=True)
    v_normalized = v / np.linalg.norm(v, axis=1, keepdims=True)
    C = 1 - np.dot(u_normalized, v_normalized.T)
    a = np.ones(len(u)) / len(u)
    b = np.ones(len(v)) / len(v)
    return ot.emd2(a, b, C)


def f_LDR(u, v):
    s_LDR, t_LDR = utils.calc_ldr(u, v)
    return np.abs(np.mean(s_LDR) - np.mean(t_LDR))


def f_APD(u, v):
    u_normalized = u / np.linalg.norm(u, axis=1, keepdims=True)
    v_normalized = v / np.linalg.norm(v, axis=1, keepdims=True)
    u_mean = np.mean(u_normalized, axis=0)
    v_mean = np.mean(v_normalized, axis=0)
    dist = 1 - u_mean@v_mean
    return dist


def f_WiDiD(u, v, damping=0.5):
    Ls, Lt = widid(u, v, damping=damping)
    return jsd(Ls, Lt)

def f_APDP(u, v, damping=0.5):
    Ls, Lt = widid(u, v, damping=damping)
    return apdp(u, v, Ls, Lt)


def g_SUS(u, v, reg_m=100):
    s_sus, t_sus = utils.calc_sus(u, v, reg_m=reg_m)
    return np.log(np.var(t_sus) / np.var(s_sus))


def g_vMF(u, v):
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
    return np.log(ks/kt)


def g_LDR(u, v):
    s_LDR, t_LDR = utils.calc_ldr(u, v)
    return np.log(np.var(t_LDR) / np.var(s_LDR))

def g_WiDiD(u, v, damping=0.5):
    Ls, Lt = widid(u, v, damping=damping)
    return entropy_diff(Ls, Lt)

def tau_SUS(u, v, reg_m=100):
    s_sus, t_sus = utils.calc_sus(u, v, reg_m=reg_m)
    return s_sus, t_sus

def tau_LDR(u, v):
    s_LDR, t_LDR = utils.calc_ldr(u, v)
    return s_LDR, t_LDR

def tau_WiDiD(u, v, damping=0.5):
    Ls, Lt = widid(u, v, damping=damping)
    labels = np.unique(np.concatenate([Ls, Lt]))
    cs = Counter(Ls)
    ct = Counter(Lt)
    Ls_dist = np.array([cs[l] for l in labels])
    Lt_dist = np.array([ct[l] for l in labels])
    Ls_dist = Ls_dist / Ls_dist.sum()
    Lt_dist = Lt_dist / Lt_dist.sum()
    cluster2tau = {i: v for i, v in enumerate(np.log(Lt_dist / Ls_dist))}
    max_val = [val for val in cluster2tau.values() if val != np.inf]
    min_val = [val for val in cluster2tau.values() if val != -np.inf]
    cluster2tau = {k: v for k, v in cluster2tau.items()}
    source_tau = np.array([cluster2tau[l] for l in Ls])
    target_tau = np.array([cluster2tau[l] for l in Lt])
    return source_tau, target_tau
    
