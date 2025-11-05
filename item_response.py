# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, Vu Van Thang (2201040169), declare that this code is my own original work.
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
    """Numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-x))


def neg_log_likelihood(data, theta, beta, lamb=0.0):
    """Compute the negative log-likelihood for the IRT model."""
    user_ids = data["user_id"]
    question_ids = data["question_id"]
    is_correct = data["is_correct"]

    log_likelihood = 0.0
    for i, q, c in zip(user_ids, question_ids, is_correct):
        x = theta[i] - beta[q]
        p = sigmoid(x)
        log_likelihood += c * np.log(p + 1e-9) + (1 - c) * np.log(1 - p + 1e-9)

    reg = (lamb / 2) * (np.sum(theta ** 2) + np.sum(beta ** 2))
    return -log_likelihood + reg


def update_theta_beta(data, lr, theta, beta, lamb=0.0):
    """Update theta and beta using gradient descent."""
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

    # Apply L2 regularization
    d_theta -= lamb * theta
    d_beta -= lamb * beta

    theta += lr * d_theta
    beta += lr * d_beta
    return theta, beta


def evaluate(data, theta, beta):
    """Evaluate model accuracy."""
    pred = []
    for i, q, c in zip(data["user_id"], data["question_id"], data["is_correct"]):
        p = sigmoid(theta[i] - beta[q])
        pred.append(p >= 0.5)
    return np.mean(np.array(pred) == np.array(data["is_correct"]))


def irt(train_data, val_data, lr, iterations, lamb=0.0):
    """Train IRT model using alternating gradient descent."""
    n_users = max(train_data["user_id"]) + 1
    n_questions = max(train_data["question_id"]) + 1

    theta = np.random.normal(0, 0.1, n_users)
    beta = np.random.normal(0, 0.1, n_questions)

    val_acc_lst, nll_list = [], []

    for i in range(iterations):
        nll = neg_log_likelihood(train_data, theta, beta, lamb)
        val_acc = evaluate(val_data, theta, beta)
        val_acc_lst.append(val_acc)
        nll_list.append(nll)

        print(f"Epoch {i+1}/{iterations} | NLL={nll:.4f} | ValAcc={val_acc:.4f}")

        theta, beta = update_theta_beta(train_data, lr, theta, beta, lamb)

    return theta, beta, val_acc_lst, nll_list


def plot_results(nll_list, val_acc_lst, student_id):
    """Plot NLL and validation accuracy over epochs."""
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


def plot_question_curves(theta, beta, question_ids, student_id):
    """Plot p(c_ij=1) vs theta for 3 selected questions."""
    theta_range = np.linspace(min(theta), max(theta), 200)
    plt.figure(figsize=(8, 5))

    for q in question_ids:
        probs = sigmoid(theta_range - beta[q])
        plt.plot(theta_range, probs, label=f"Question {q}")

    plt.title("IRT Probability Curves for Selected Questions")
    plt.xlabel("Student ability (θ)")
    plt.ylabel("P(correct | θ, β)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"irt_curves_{student_id}.png")
    plt.close()


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    best_acc = 0
    best_params = None

    for lr in [0.01, 0.05]:
        for lamb in [0.0, 0.01]:
            for it in [20, 50]:
                theta, beta, val_acc, nll = irt(train_data, val_data, lr, it, lamb)
                acc = val_acc[-1]
                if acc > best_acc:
                    best_acc = acc
                    best_params = (lr, lamb, it)
                    best_theta, best_beta = theta, beta
                    best_val_acc, best_nll = val_acc, nll

    test_acc = evaluate(test_data, best_theta, best_beta)
    print(f"Best Params: lr={best_params[0]}, λ={best_params[1]}, iter={best_params[2]}")
    print(f"Validation Acc={best_acc:.4f}, Test Acc={test_acc:.4f}")

    student_id = "2201040169"
    plot_results(best_nll, best_val_acc, student_id)

    # Visualization part (d)
    selected_questions = [0, 100, 500]  # example question IDs
    plot_question_curves(best_theta, best_beta, selected_questions, student_id)


if __name__ == "__main__":
    main()
