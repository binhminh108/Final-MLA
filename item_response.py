# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, [Student Name] ([Student ID]), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, lamb=0.0):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_ids = data["user_id"]
    question_ids = data["question_id"]
    is_correct = data["is_correct"]

    log_likelihood = 0.0
    for i, q, c in zip(user_ids, question_ids, is_correct):
        x = theta[i] - beta[q]
        p = sigmoid(x)
        log_likelihood += c * np.log(p + 1e-9) + (1 - c) * np.log(1 - p + 1e-9)

    # add L2 regularization
    reg = (lamb / 2) * (np.sum(theta**2) + np.sum(beta**2))
    #####################################################################
    #                               END OF YOUR CODE                    #
    #####################################################################
    return -log_likelihood + reg


def update_theta_beta(data, lr, theta, beta, lamb=0.0):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_ids = data["user_id"]
    question_ids = data["question_id"]
    is_correct = data["is_correct"]

    d_theta = np.zeros_like(theta)
    d_beta = np.zeros_like(beta)

    for i, q, c in zip(user_ids, question_ids, is_correct):
        x = theta[i] - beta[q]
        p = sigmoid(x)
        d_theta[i] += (c - p)
        d_beta[q] += -(c - p)

    # L2 regularization
    d_theta -= lamb * theta
    d_beta -= lamb * beta

    theta += lr * d_theta
    beta += lr * d_beta
    #####################################################################
    #                               END OF YOUR CODE                    #
    #####################################################################
    return theta, beta


def irt(train_data, val_data, lr, iterations, lamb=0.0):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    #####################################################################
    # TODO: Initialize theta and beta.
    n_users = max(train_data["user_id"]) + 1
    n_questions = max(train_data["question_id"]) + 1

    theta = np.random.normal(0, 0.1, n_users)
    beta = np.random.normal(0, 0.1, n_questions)
    #####################################################################
    #                               END OF YOUR CODE                    #
    #####################################################################

    val_acc_lst = []
    nll_list = []

    for i in range(iterations):
        nll = neg_log_likelihood(train_data, theta, beta, lamb)
        score = evaluate(val_data, theta, beta)
        val_acc_lst.append(score)
        nll_list.append(nll)

        print(f"Epoch {i+1}/{iterations} | NLL={nll:.4f} | ValAcc={score:.4f}")
        theta, beta = update_theta_beta(train_data, lr, theta, beta, lamb)

    #####################################################################
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, nll_list
    #####################################################################
    #                               END OF YOUR CODE                    #
    #####################################################################


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])

def plot_results(nll_list, val_acc_lst, student_id):
    fig, ax1 = plt.subplots()
    ax1.plot(range(len(nll_list)), nll_list, label='Training NLL', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='blue')
    ax2 = ax1.twinx()
    ax2.plot(range(len(val_acc_lst)), val_acc_lst, label='Validation Accuracy', color='red')
    ax2.set_ylabel('Validation Accuracy', color='red')
    plt.title('IRT Training Progress')
    plt.savefig(f"irt_results_{student_id}.png")
    plt.close()

def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    best_acc = 0
    best_params = None
    results = {}

    
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    for lr in [0.001, 0.01, 0.05]:
        for lamb in [0.0, 0.01, 0.1]:
            for it in [20, 50]: 
                theta, beta, val_acc, nll = irt(train_data, val_data, lr, it, lamb)
                acc = val_acc[-1]
                results[(lr, lamb, it)] = acc
                if acc > best_acc:
                    best_acc = acc
                    best_params = (lr, lamb, it)
                    best_theta, best_beta, best_val_acc, best_nll = theta, beta, val_acc, nll

    test_acc = evaluate(test_data, best_theta, best_beta)
    print(f"Best Params: lr={best_params[0]}, λ={best_params[1]}, iter={best_params[2]}")
    print(f"Validation Acc={best_acc:.4f}, Test Acc={test_acc:.4f}")

    plot_results(best_nll, best_val_acc, student_id="123456")
    #####################################################################
    #                               END OF YOUR CODE                    #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                               END OF YOUR CODE                    #
    #####################################################################


if __name__ == "__main__":
    main()