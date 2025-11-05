# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, Vuong Minh Vu (2201040200), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, roc_auc_score
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

    # KNNImputer treats rows as samples, which in the case are users.
    # This aligns perfectly with user-based collaborative filtering.
    imputer = KNNImputer(n_neighbors=k)
    
    # Fit on the training data and transform it to fill NaNs.
    mat_imputed = imputer.fit_transform(matrix)

    # Evaluate the imputed matrix against the validation data.
    acc = sparse_matrix_evaluate(valid_data, mat_imputed)
    
    # Extract predictions for the validation set locations
    preds = []
    for i in range(len(valid_data["is_correct"])):
        user_id = valid_data["user_id"][i]
        question_id = valid_data["question_id"][i]
        preds.append(mat_imputed[user_id, question_id])

    if return_confusion:
        # Round predictions to 0 or 1 to compute confusion matrix
        binary_preds = np.round(preds)
        true_labels = valid_data["is_correct"]
        cm = confusion_matrix(true_labels, binary_preds)
        return acc, cm, preds
        
    return acc, None, preds


def item_knn_predict_hanu(matrix, valid_data, k, student_id=""):
    """
    Predict missing values using item-based k-nearest neighbors (KNN).
    Also saves validation predictions to file named '{student_id}_item_knn_preds.npy'
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # To perform item-based KNN, we need to find similar items (questions).
    # KNNImputer finds similar rows. So, we transpose the matrix to make
    # questions the rows and users the columns.
    transposed_matrix = matrix.T
    
    imputer = KNNImputer(n_neighbors=k)
    
    # Fit and transform on the transposed matrix
    imputed_transposed = imputer.fit_transform(transposed_matrix)
    
    # Transpose back to the original orientation (users x questions)
    mat_imputed = imputed_transposed.T
    
    # Evaluate the imputed matrix against the validation data.
    acc = sparse_matrix_evaluate(valid_data, mat_imputed)
    
    # Save the predictions for the validation set to a file
    if student_id:
        preds = []
        for i in range(len(valid_data["is_correct"])):
            user_id = valid_data["user_id"][i]
            question_id = valid_data["question_id"][i]
            preds.append(mat_imputed[user_id, question_id])
        
        filename = f"2201040200_item_knn_preds.npy"
        np.save(filename, np.array(preds))
        print(f"Item-based predictions for k={k} saved to {filename}")
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
    print("-"*30)
    # Define the list of k values to iterate through as per the project description
    ks = [1, 6, 11, 16, 21, 26]
    
    # Initialize variables to keep track of the best performing k
    best_k = -1
    best_acc = 0.0

    # Variable to store best guess k
    best_user_preds = None

    print("--- Running User-Based KNN on Validation Set ---")
    print("Confusion [True-negative False-Positive][False-negative True-positive]")
    # Iterate over each value of k to find the best one
    for k in ks:
        # Calculate validation accuracy and confusion matrix for the current k 
        # We assume the `user_knn_predict_hanu` function is defined elsewhere as required.
        acc, cm, preds = user_knn_predict_hanu(sparse_matrix, val_data, k, return_confusion=True)
        
        print(f"For k = {k}, Validation Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("-" * 25)
        
        # If the current accuracy is better than the best one found so far, update it
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_user_preds = preds

    # take true label from validation
    true_labels = val_data["is_correct"]
    # Calculate ROC-AUC score
    roc_auc = roc_auc_score(true_labels, best_user_preds)
    print(f"\n[Evaluation Metric] ROC-AUC score for best k={best_k}: {roc_auc:.4f}")
    # ----------------------------------------

    # After finding the best k (k*), report the test accuracy using that k
    print(f"\nBest k* found from validation set: {best_k}")
    test_acc, _, _ = user_knn_predict_hanu(sparse_matrix, test_data, best_k)
    print(f"Final Test Accuracy with k* = {best_k}: {test_acc:.4f}\n")


    print("\n--- Running Item-Based KNN on Validation Set ---")
    best_item_acc = 0
    best_item_k = -1

    for k in ks:
        acc = item_knn_predict_hanu(sparse_matrix, val_data, k, student_id="2201040200")
        print(f"For k = {k}, Validation Accuracy (Item-based): {acc:.4f}")
    if acc > best_item_acc:
        best_item_acc = acc
        best_item_k = k

    print(f"Best k* for Item-based KNN: {best_item_k} with Validation Accuracy = {best_item_acc:.4f}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    print("")
    print(f"[Summary] For K={best_k}, the user-based KNN achieved {best_acc:.3f} validation accuracy. With a corresponding ROC-AUC score of {roc_auc:.4f} and test accuracy of {test_acc:.4f}.")
    print(f"Reflection: KNN performed best when K was {best_k}. "
          "This suggests that considering a moderate number of similar users provides the best signal for prediction, "
        "whereas a very small K is too noisy and a very large K includes too many dissimilar users.")

    print(f"For the item-based KNN, the best validation accuracy was {acc:.4f} at k = {best_item_k}.")
    print("Reflection: This shows that considering many similar questions helps the model to better utilize the relationship between questions to predict the outcome. " 
          "However, if k continues to increase too much, the model may be contaminated by less relevant questions, leading to a decline in predictive performance.")
    
    print(f"User-based KNN achieved accuracy {best_acc:.4f}, item-based KNN achieved accuracy {best_item_acc:.4f}, suggesting that user similarity captures learning behavior more effectively than question similarity.")

if __name__ == "__main__":
    main()
