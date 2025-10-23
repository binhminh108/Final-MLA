# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, [Student Name] ([Student ID]), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import load_valid_csv, load_public_test_csv, load_train_sparse

def load_data(base_path="./data"):
    """
    Load the data and return: zero_train_matrix, train_matrix, valid_data, test_data.
    zero_train_matrix: missing entries filled with 0 (for input).
    train_matrix: preserves NaNs (for masking).
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)
    return zero_train_matrix, train_matrix, valid_data, test_data

class AutoEncoder(nn.Module):
    """
    Simple autoencoder for student response prediction.
    - Input: response vector (missing as 0)
    - Encoder: Linear + sigmoid
    - Decoder: Linear + sigmoid
    """
    def __init__(self, num_question, k=100):
        super(AutoEncoder, self).__init__()
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2 for regularization."""
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """
        TODO:
        - Implement the forward pass:
          out = sigmoid(h(sigmoid(g(inputs))))
        - Return the output vector.
        """
        # === YOUR CODE HERE ===
        raise NotImplementedError("Student must implement the autoencoder forward pass.")
        # return out

def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, student_id=""):
    """
    Train the autoencoder with L2 regularization.
    TODO:
    - For each epoch, for each user:
        * Compute forward pass.
        * Compute masked squared error loss for observed entries only.
        * Add L2 regularization (lamb * model.get_weight_norm()).
        * Backprop and optimizer step.
    - Track/plot training loss and validation accuracy.
    - Save plot as autoencoder_results_{student_id}.png.
    - Print summary with best k/lambda/validation accuracy.
    """
    # === YOUR CODE HERE ===
    raise NotImplementedError("Student must implement the training loop with regularization and tracking.")

def evaluate(model, train_data, valid_data):
    """
    Evaluate the model on valid_data. (Already provided.)
    """
    model.eval()
    total = 0
    correct = 0
    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)

def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    num_question = train_matrix.shape[1]

    #####################################################################
    # TODO:
    # 1. Try at least 5 values of k; select best k via validation set.
    # 2. Tune lr, lamb, num_epoch.
    # 3. Train AutoEncoder, plot/save learning curves, validation accuracy.
    # 4. Report best k and corresponding metrics.
    # 5. Save plot as autoencoder_results_{student_id}.png.
    # 6. Write a reflection on regularization and k in your report.
    #####################################################################

    # Example (students must replace with their chosen values)
    k = None     # e.g. 10, 50, 100, 200, 500
    lr = None
    num_epoch = None
    lamb = None
    student_id = "studentID"

    model = AutoEncoder(num_question, k)
    train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, student_id)
    # After training, evaluate on validation and/or test data as required.

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

if __name__ == "__main__":
    main()
