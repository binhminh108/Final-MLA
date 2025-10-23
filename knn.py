# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, [Student Name] ([Student ID]), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def user_knn_predict_hanu(matrix, valid_data, k, return_confusion=False):
    """
    Predict missing values using user-based k-nearest neighbors (KNN).
    Args:
        matrix: 2D numpy array (users x questions) with NaNs for missing.
        valid_data: dict with user_id, question_id, is_correct.
        k: int, number of neighbors.
        return_confusion: bool, if True also return confusion matrix.
    Returns:
        accuracy: float
        (optional) confusion_matrix: sklearn confusion matrix
    """
    # Implementation...


def item_knn_predict_hanu(matrix, valid_data, k, student_id=""):
    """
    Predict missing values using item-based k-nearest neighbors (KNN).
    Also saves validation predictions to file named '{student_id}_item_knn_preds.npy'
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    acc = None
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    print(f"[Summary] For K={best_k}, the user-based KNN achieved {best_acc:.3f} validation accuracy.")
    print("Reflection: KNN performed best when K was ... [student must edit].")


if __name__ == "__main__":
    main()
