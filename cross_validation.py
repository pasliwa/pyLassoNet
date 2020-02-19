import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import data_generation as dg
import design_matrix as dm
from scipy.interpolate import UnivariateSpline
import quadratic_optimization as qo
import sys


where_to_save = "filepath"
model_name ="Predator_prey_tp.txt"
res = 2.20
mean = 0
scale= 1.0


def discard(X, perc):
    start = int(X.shape[0] * perc)
    end = X.shape[0] - start
    return X[start:end]

def discard_time(times, perc):
    start = int(times.shape[0] * perc)
    end = times.shape[0] - start
    return times[start:end] - times[start]

def spline_ground(X_dot_eq, splined_X_dot, ax, alpha):
    plt.sca(ax)
    plt.scatter(X_dot_eq[:, 0], splined_X_dot[:, 0] - X_dot_eq[:, 0], 1, alpha=alpha)
    plt.scatter(X_dot_eq[:, 1], splined_X_dot[:, 1] - X_dot_eq[:, 1], 1, alpha=alpha)
    plt.xlabel("X_dot", fontdict={"size": 20})
    plt.ylabel("splined_X_dot - X_dot", fontdict={"size": 20})
    plt.title("Spline vs ground-truth", fontdict={"size": 25})
    ax.axis('equal')
    return

def spline_ground_X_axis(X_dot_eq, splined_X_dot, ax, alpha, X_axis, X_axis_name="X_dot_dot"):
    plt.sca(ax)
    plt.scatter(X_axis[:, 0], splined_X_dot[:, 0] - X_dot_eq[:, 0], 1, alpha=alpha)
    plt.scatter(X_axis[:, 1], splined_X_dot[:, 1] - X_dot_eq[:, 1], 1, alpha=alpha)
    plt.xlabel(X_axis_name, fontdict={"size": 20})
    plt.ylabel("splined_X_dot - X_dot", fontdict={"size": 20})
    plt.title("Spline vs ground-truth", fontdict={"size": 25})
    ax.axis('equal')
    return


labels, init_conc, S_matrix, educt_matrix, kin_par, t_T, t_eval_step = dg.read_model(model_name)
t_eval_step = res

prop_E_matrix, prop_S_matrix = dm.proposed_functions(2, max_sum_educts=2, max_sum_products=3,
                                                     max_product_val=2,
                                                     max_absolute_stoichiometric=1)

times, clean_X, X_dot_eq, noisy_X = generate_data(init_conc, S_matrix, educt_matrix, kin_par, t_T, t_eval_step, mean, scale)
X_dot_dot_eq = give_second_derivative_vec_PP(clean_X, kin_par)
noisy_X, splined_X, splined_X_dot, splined_X_dot_dot = give_res_scale_i(data, res, scale, i, perc=0.05)

data = np.load(os.path.join(where_to_save, f"simulation_{res:.2f}_{scale:.2f}_{iden}.npy"), allow_pickle=True)
noisy_X, splined_X, splined_X_dot, splined_X_dot_dot = give_res_scale_i(data, res, scale, i, perc=0.05)
data_c = np.load(os.path.join(where_to_save, f"simulation_non_noisy_{res:.2f}_{iden}.npy"), allow_pickle=True)

times, X, X_dot_eq = give_res(data_c, res, perc=0.05)
