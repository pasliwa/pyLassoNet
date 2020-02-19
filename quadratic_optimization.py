import cvxopt
import numpy as np
from scipy import sparse

cvxopt.solvers.options['show_progress'] = False


def give_flattened(X_dot, theta_matrix):
    """
    Flattens X_dot and theta_matrix according to Fortran
    maintaining correct ordering of the data (so later multiplication makes sense)
    """
    return X_dot.reshape((-1), order="F"), theta_matrix.reshape((-1, theta_matrix.shape[-1]), order="F")

def add_lasso_t_constraint(non_negative_LHS, non_negative_RHS, lasso_t):
    return np.vstack([non_negative_LHS, np.ones(non_negative_LHS.shape[0])]), np.hstack([non_negative_RHS, lasso_t])



def give_qp_input(X_dot, theta_matrix, w=None):
    """
    Formats the data so, that it is accepted by cvxopt
    H, f correspond to notation from Daniel's report
    """
    X_dot_, theta_matrix_ = give_flattened(X_dot, theta_matrix)
    if w is None:
        w = np.ones(X_dot_.shape[0])
    T = X_dot.shape[0]
    H = (theta_matrix_ * w[:, np.newaxis]).T.dot(theta_matrix_)
    ft = -(X_dot_ * w).T.dot(theta_matrix_)  # remember about the minus
    num_ansatz = theta_matrix_.shape[-1]
    non_negative_LHS = -np.identity(num_ansatz)
    lasso_LHS = np.ones(num_ansatz)
    non_negative_RHS = np.zeros(num_ansatz)
    return H, ft.T, non_negative_LHS, non_negative_RHS


def give_error(theta_matrix, sol, X_dot):
    """
    Calculates the LSE
    That is normalized Frobenius norm of squared distances between X_dot estimated from solution (and theta) and true X_dot
    :param theta_matrix:
    :param sol:
    :param X_dot:
    :return:
    """

    def estimate_derivative(theta_matrix, xi_vector):
        return np.dot(theta_matrix, xi_vector)

    estimated_X_dot = estimate_derivative(theta_matrix, np.array(list(sol["x"])))
    return (np.linalg.norm(X_dot - estimated_X_dot) ** 2) / (2 * np.prod(X_dot.shape))


def run_qp(X_dot, theta_matrix, lasso_t, w=None):
    """
    Run quadratic program for X_dot, theta_matrix at given lasso_t constraint
    Returns sol - dictionary, whose most important keys are:
    "x" - cvxopt.base.matrix, can be changed into python list by list(sol["x"])
    "status" - string information whether optimal state was reached
    "primal objective" - float with primal objective function value (mind the transformation,
    gives value proportional to the error [differing by a constant X_dot.T * X_dot / 2T])
    """
    if w is not None:
        if len(X_dot.shape) > 1:
            w = np.tile(w, X_dot.shape[1])
    P, q, non_negative_LHS, non_negative_RHS = give_qp_input(X_dot, theta_matrix, w)
    G, h = add_lasso_t_constraint(non_negative_LHS, non_negative_RHS, lasso_t)
    P = cvxopt.matrix(P, tc='d')
    q = cvxopt.matrix(q, tc='d')
    G = cvxopt.matrix(G, tc='d')
    h = cvxopt.matrix(h, tc='d')
    sol = cvxopt.solvers.qp(P, q, G, h)
    return sol


def lasso_qp_path(X_dot, theta_matrix, lasso_ts, X_dot_val, theta_matrix_val,  w=None):
    if w is not None:
        if len(X_dot.shape) > 1:
            w = np.tile(w, X_dot.shape[1])
    P, q, non_negative_LHS, non_negative_RHS = give_qp_input(X_dot, theta_matrix, w)
    G = np.vstack([non_negative_LHS, np.ones(non_negative_LHS.shape[0])])
    P = cvxopt.matrix(P, tc='d')
    q = cvxopt.matrix(q, tc='d')
    G = cvxopt.matrix(G, tc='d')
    initvals= None
    alphas = []
    coefs = []
    errors_train = []
    errors_validation = []
    for lasso_t in lasso_ts:
        h =  np.hstack([non_negative_RHS, lasso_t])
        h = cvxopt.matrix(h, tc='d')
        sol = cvxopt.solvers.qp(P, q, G, h, initvals=initvals)
        #print(sol["status"])
        if sol["status"] != "optimal":
            raise NameError("Error - not optimal")
        initvals = dict()
        initvals["x"] = sol["x"]
        #initvals["s"] = sol["s"]
        #initvals["y"] = sol["y"]
        #initvals["z"] = sol["z"]
        alphas.append(lasso_t)
        coefs.append(np.array(list(sol["x"])))
        errors_train.append(give_error(theta_matrix, sol, X_dot))
        errors_validation.append(give_error(theta_matrix_val, sol, X_dot_val)) #train_index
        # test_errors
    return alphas, coefs, errors_train, errors_validation
