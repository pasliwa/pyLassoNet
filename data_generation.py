#!/usr/bin/env python3

"""
Written by: Piotr Sliwa
Programming language: python-3.7.3
Used libraries: scipy-1.1.0, numpy-1.16.3, matplotlib-3.0.3
Tested on: Ubuntu 18.04.2
"""

import numpy as np
from scipy.integrate import solve_ivp
from functools import partial



def calculate_fluxes(conc, S_matrix, educt_matrix, kin_par):
    """
    Calculates reaction rates according to formula from lecture
    R_j(conc) = k_j * product_{i=1}^{n_s}(conc_i^{educt_{ij}})

    :param conc: concentrations
    :param S_matrix: stoichiometric matrix (product_{ij} - educt_{ij})
    :param educt_matrix: educt matrix
    :param kin_par: kinetic parameters
    :return: reaction rates for given concentrations (under mass-action kinetics)
    """

    educts_per_reaction = educt_matrix.T

    # in every reaction j row, element at pos (column, species) i equals conc_i^educt_{ij}
    # educt_{ij} - how many units of i are educts in reaction j
    multiplicative_term_rows = np.power(conc, educts_per_reaction)

    # concentrations to educt product    ---->    product_{i=1}^{n_s}(conc_i^{educt_{ij}}
    concentration_product = np.prod(multiplicative_term_rows, axis=1)

    # returns one row with #reactions elements -> fluxes
    return kin_par * concentration_product


def theta(conc, S, educts):
    """
    Calculates reaction rates according to formula from lecture
    R_j(conc) = k_j * product_{i=1}^{n_s}(conc_i^{educt_{ij}})

    :param conc: concentrations
    :param S_matrix: stoichiometric matrix (product_{ij} - educt_{ij})
    :param educt_matrix: educt matrix
    :param kin_par: kinetic parameters
    :return: reaction rates for given concentrations (under mass-action kinetics)
    """

    educts_per_reaction = educts.T

    # in every reaction j row, element at pos (column, species) i equals conc_i^educt_{ij}
    # educt_{ij} - how many units of i are educts in reaction j
    multiplicative_term_rows = np.power(conc, educts_per_reaction)

    # concentrations to educt product    ---->    product_{i=1}^{n_s}(conc_i^{educt_{ij}}
    concentration_product = np.prod(multiplicative_term_rows, axis=1)

    # returns one row with #reactions elements -> fluxes
    return concentration_product * S


def rhs(t, conc, S_matrix, educt_matrix, kin_par):
    """ Calculate right hand side """
    fluxes = calculate_fluxes(conc, S_matrix, educt_matrix, kin_par)
    return np.dot(S_matrix, fluxes)

def generate_X(init_conc, S_matrix, educt_matrix, kin_par, times, t_T):

    rhs_wr = partial(rhs, S_matrix=S_matrix, educt_matrix=educt_matrix, kin_par=kin_par)

    sol = solve_ivp(fun=rhs_wr, t_span=[0, t_T], y0=init_conc, t_eval=times)

    # X belongs to R_{0, +}^{T x species}
    X = sol.y.T
    return X


def give_clean_data_from_eq(init_conc, S_matrix, educt_matrix, kin_par, t_eval_step, t_T):
    times = np.arange(0, t_T, t_eval_step)
    X = generate_X(init_conc, S_matrix, educt_matrix, kin_par, times, t_T)
    rhs_s = []
    for k, conc in enumerate(X):
        rhs_s.append(rhs(k, conc, S_matrix, educt_matrix, kin_par))
    X_dot_eq = np.stack(rhs_s)
    return times, X, X_dot_eq


def give_noisy_data(X, mean, scale):
    return X + np.random.normal(mean, scale, X.shape)


def read_model(input_file):
    """ Read data in format specified above """
    with open(input_file) as inp:
        labels = inp.readline().strip().split(" ")
        init_conc = np.array(list(map(float, inp.readline().strip().split(" "))))

        stoich = []
        for i in range(len(labels)):
            stoich.append(list(map(float, inp.readline().strip().split(" "))))
        S_matrix = np.array(stoich)

        educt = []
        for i in range(len(labels)):
            educt.append(list(map(float, inp.readline().strip().split(" "))))
        educt_matrix = np.array(educt)

        kin_par = np.array(list(map(float, inp.readline().strip().split(" "))))
        t_T, t_eval_step = list(map(float, inp.readline().strip().split(" ")))

    return labels, init_conc, S_matrix, educt_matrix, kin_par, t_T, t_eval_step

def generate_data(init_conc, S_matrix, educt_matrix, kin_par, t_T, t_eval_step, mean, scale):
    times, clean_X, X_dot_eq = give_clean_data_from_eq(init_conc, S_matrix, educt_matrix, kin_par, t_eval_step, t_T)
    noisy_X = give_noisy_data(clean_X, mean, scale)
    return times, clean_X, X_dot_eq, noisy_X


def generate_data_from_file(file_name, mean, scale):
    labels, init_conc, S_matrix, educt_matrix, kin_par, t_T, t_eval_step = read_model(file_name)
    times, clean_X, X_dot_eq = give_clean_data_from_eq(init_conc, S_matrix, educt_matrix, kin_par, t_eval_step)
    noisy_X = give_noisy_data(clean_X, mean, scale)
    return times, clean_X, X_dot_eq, noisy_X

def give_second_derivative_vec_PP(x, k):
    ddx1 = k[0]**2 * x[:, 0] + (k[3] - 2 * k[0]) * (k[1] + k[2]) * x[:, 0] * x[:, 1] + (k[1] + k[2])**2 * x[:, 0] * x[:, 1]**2 - k[2] * (k[1] + k[2]) * x[:, 0]**2 * x[:, 1]
    ddx2 = k[2] * (k[0] - 2 * k[3]) * x[:, 0] * x[:, 1] + k[3]**2 * x[:, 1] - k[2] * (k[1] + k[2]) * x[:, 0] * x[:, 1] **2 + k[2]**2 * x[:, 0]**2 * x[:, 1]
    return np.array([ddx1, ddx2]).T
