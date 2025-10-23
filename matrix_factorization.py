# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, [Student Name] ([Student ID]), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)

def svd_reconstruct_hanu(matrix, k):
    """
    Given the matrix, perform SVD to reconstruct the matrix using the top k components.
    - Fill missing values (e.g., with column mean).
    - Center the matrix.
    - Perform SVD and reconstruct using k components.
    Returns: reconstructed matrix (n_users x n_questions).
    """
    # TODO: Implement as described above.
    raise NotImplementedError("Student must implement SVD-based matrix reconstruction.")

def squared_error_loss(data, u, z, lambda_=0.0):
    """
    Compute squared-error loss WITH L2 regularization given the data.
    Returns: float (total loss)
    """
    # TODO: Implement regularized loss.
    # Loss = (1/2) sum_{(n,m)} (c_nm - u_n^T z_m)^2 + (lambda_/2)(||U||^2 + ||Z||^2)
    raise NotImplementedError("Student must implement regularized squared error loss.")

def update_u_z(train_data, lr, u, z, lambda_=0.0):
    """
    Perform a SGD update with L2 regularization.
    TODO:
    - Randomly pick an observed (user, question) pair.
    - Compute gradients including L2 terms and update u, z.
    Returns: updated u, z
    """
    raise NotImplementedError("Student must implement update step for ALS.")

def als(train_data, valid_data, k, lr, num_iteration, lambda_=0.01, student_id=""):
    """
    ALS with SGD and L2 regularization.
    TODO:
    - Initialize u, z.
    - For each iteration:
        * Call update_u_z (enough times to touch all observed entries).
        * Track and plot both training loss and validation accuracy (use valid_data).
    - Save results using student_id.
    Returns: predicted matrix (u @ z.T)
    """
    raise NotImplementedError("Student must implement ALS main loop.")

def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # SVD: Experiment with at least 5 k, report val accuracy, select best
    #####################################################################
    # TODO: Your SVD experiments here.
    # For each k, reconstruct, evaluate on val_data, and save/report results.
    # Plot/compare SVD vs ALS.
    pass

    #####################################################################
    # ALS: Experiment with at least 5 k, report val accuracy, select best
    #####################################################################
    # TODO: Your ALS experiments here.
    # For each k, train ALS, evaluate, plot/save learning curves as mf_results_{student_id}.png.
    # Save U/Z matrices with student_id in filename.
    # Plot/compare SVD vs ALS.
    pass

    #####################################################################
    # Reflection:
    # In your report, discuss:
    # - Hyperparameter tuning process and validation
    # - Comparison of SVD and ALS
    # - Limitations of each method (esp. SVD w.r.t missing data)
    # - Effect of regularization (lambda)
    # - Plots and tables as required by assignment
    #####################################################################

    print(f"Summary: With k={{best_k}} and lambda={{lambda_}}, the validation accuracy was ... [student edits]. Main plot saved as mf_results_{{student_id}}.png.")

if __name__ == "__main__":
    main()
