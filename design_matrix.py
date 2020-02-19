import itertools
import os
import pickle
import sys
from functools import partial

import cvxopt
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline


where_to_save = "where_to_save"
# from sklearn.model_selection import ParameterGrid




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

    # returns #species rows with #reactions elements -> influences on given variable coming from each of the reactions
    return concentration_product * S



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
        t_T, tau = list(map(float, inp.readline().strip().split(" ")))

    return labels, init_conc, S_matrix, educt_matrix, kin_par, t_T, tau


def ij(max_sum):
    """
    All i, j that sum up to values <= sum_max (i + j <= sum_max)
    :param max_sum:
    :return:
    """
    for i in range(max_sum + 1):
        for j in range(max_sum + 1):
            if (i + j <= max_sum) and (i + j != 0):
                yield (i, j)


def all_educts(num_species, max_sum_educts=2):
    """
    Return all possible educt values
    that fulfill 1) condition [sum(educts) <= 2]
    :param num_species:
    :param maximal_stoichiometry: usually assumed to be 2
    :return:
    """
    return np.array([i for i in list(itertools.product(range(0, max_sum_educts + 1), repeat=num_species)) if
                     sum(i) <= max_sum_educts])


def give_stoichs(species, abs_sum_stoich_max=1):
    """
    Give stoichiometric matrices fulfilling stoichiometric condition
    :param species:
    :param abs_sum_stoich_max:
    :return:
    """
    for plus_ones in range(2 + 1):
        for minus_ones in range(2 + 1):
            for plus_two in range(1 + 1):
                for minus_two in range(1 + 1):
                    S_condition = (abs(
                        (-2 * minus_two) + (2 * plus_two) + (-1 * minus_ones) + (1 * plus_ones)) <= abs_sum_stoich_max)
                    species_condition = ((species - (plus_ones + minus_ones + plus_two + minus_two)) >= 0)
                    not_all_zeros = ((plus_ones + minus_ones + plus_two + minus_two) != 0)
                    if S_condition and species_condition and not_all_zeros:  # S condition
                        yield list(set(itertools.permutations(
                            np.hstack((np.ones(plus_ones), -np.ones(minus_ones), 2 * np.ones(plus_two),
                                       -2 * np.ones(minus_two),
                                       np.zeros(species - (plus_ones + minus_ones + plus_two + minus_two)))))))


def give_flat_stoichs(n, max_absolute_stiochiometric=1):
    """
    All possible changes by at most 1 (element wise) for n elements

    :param n: number of elements
    :return: flat numpy array
    """
    return np.array(
        [item for sublist in list(give_stoichs(n)) for item in sublist])


def products_from_educt_stoichs(educt, stoichs, max_sum_products=3, max_product_val=2):
    """
    Non-negative possible products
    fulfilling condition 2) of stoichiometric changes [sum(products) <= 2]
    :return:
    """
    possibilities = []
    for prod in educt + stoichs:
        if np.all(prod >= 0) and (np.sum(prod) <= max_sum_products) and np.all(prod <= max_product_val):
            possibilities.append(prod)
    return np.array(possibilities)


def ones_(n):
    """
    All possible changes by at most 1 (element wise) for n elements
    :param n: number of elements
    :return:
    """
    for plus, minus in ij(n):
        yield list(
            set(itertools.permutations(np.hstack((np.ones(plus), -np.ones(minus), np.zeros(n - (plus + minus)))))))


def proposed_functions(num_species, max_sum_educts=2, max_sum_products=3, max_product_val=2,
                       max_absolute_stoichiometric=1):
    educts = all_educts(num_species, max_sum_educts)
    stoichs = give_flat_stoichs(num_species, max_absolute_stoichiometric)
    E_list = []
    S_list = []
    for educt in educts:
        products = products_from_educt_stoichs(educt, stoichs, max_sum_products, max_product_val)
        num_poss_reac = len(products)
        stoich = products - educt
        S_list.append(stoich)
        E_list.append(np.tile(educt, [num_poss_reac, 1]))

    E_matrix = np.vstack(E_list).T
    S_matrix = np.vstack(S_list).T
    return E_matrix, S_matrix


def reactions_string(E_matrix, S_matrix):
    """
    Generate readable string with reactions
    :param E_matrix:
    :param S_matrix:
    :return:
    """
    lines = ""
    last_same = False
    last = np.array([-500] * E_matrix.shape[0])
    for reaction_index in range(E_matrix.shape[1]):
        last_same = np.all(last == E_matrix[:, reaction_index])
        last = E_matrix[:, reaction_index]

        if not last_same:
            lines += "\n\n---------" + str(E_matrix[:, reaction_index]) + "---------\n\n"

        lines += "R" + str(reaction_index) + ": " + (
                str(E_matrix[:, reaction_index]) + "\t --- " + str(S_matrix[:, reaction_index]) + "\t--->\t " +
                str(E_matrix[:, reaction_index] + S_matrix[:, reaction_index])) + "\n"
    return lines

def give_theta(X, max_sum_educts=2, max_sum_products=3, max_product_val=2,
                       max_absolute_stoichiometric=1):
    if len(X.shape) > 1:
        num_species = X.shape[1]
    else:
        num_species = 1

    num_time_points = X.shape[0]
    prop_E_matrix, prop_S_matrix = proposed_functions(num_species, max_sum_educts, max_sum_products, max_product_val,
                       max_absolute_stoichiometric)

    thetas = []
    for time_point_index in range(num_time_points):  # 0 to T
        thetas.append(theta(X[time_point_index], prop_S_matrix, prop_E_matrix))
    theta_matrix = np.stack(thetas)
    return theta_matrix

def give_theta_prop(X, prop_E_matrix, prop_S_matrix):
    if len(X.shape) > 1:
        num_species = X.shape[1]
    else:
        num_species = 1

    num_time_points = X.shape[0]

    thetas = []
    for time_point_index in range(num_time_points):  # 0 to T
        thetas.append(theta(X[time_point_index], prop_S_matrix, prop_E_matrix))
    theta_matrix = np.stack(thetas)
    return theta_matrix
