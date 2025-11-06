# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, Pham Binh Minh (2201040114), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.

import numpy as np
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
)

# --- Các hàm được sao chép từ item_response.py ---
# Chúng ta cần chúng ở đây để huấn luyện các mô hình cơ sở (base models).

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
        # Đảm bảo index không vượt quá giới hạn
        if i < len(theta) and q < len(beta):
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
        if i < len(theta) and q < len(beta):
            x = theta[i] - beta[q]
            p = sigmoid(x)
            d_theta[i] += (c - p)
            d_beta[q] += -(c - p)

    # Áp dụng L2 regularization
    d_theta -= lamb * theta
    d_beta -= lamb * beta

    theta += lr * d_theta
    beta += lr * d_beta
    return theta, beta


def irt(train_data, val_data, lr, iterations, n_users, n_questions, lamb=0.0):
    """
    Train IRT model.
    SỬA ĐỔI: Chấp nhận n_users và n_questions để đảm bảo kích thước mảng chính xác.
    """
    # Khởi tạo theta và beta với kích thước đầy đủ
    theta = np.random.normal(0, 0.1, n_users)
    beta = np.random.normal(0, 0.1, n_questions)

    for i in range(iterations):
        nll = neg_log_likelihood(train_data, theta, beta, lamb)
        # Bỏ qua in ấn để quá trình ensemble nhanh hơn
        # print(f"Epoch {i+1}/{iterations} | NLL={nll:.4f}")
        theta, beta = update_theta_beta(train_data, lr, theta, beta, lamb)

    return theta, beta

# --- Kết thúc các hàm từ item_response.py ---


def bootstrap_data(data):
    """
    Tạo một mẫu bootstrap (lấy mẫu có thay thế) từ dữ liệu huấn luyện.
    """
    num_samples = len(data["user_id"])
    
    # Lấy ngẫu nhiên các chỉ số (indices) với sự thay thế
    indices = np.random.choice(num_samples, num_samples, replace=True)
    
    boot_data = {"user_id": [], "question_id": [], "is_correct": []}
    
    for i in indices:
        boot_data["user_id"].append(data["user_id"][i])
        boot_data["question_id"].append(data["question_id"][i])
        boot_data["is_correct"].append(data["is_correct"][i])
        
    return boot_data


def ensemble_evaluate(data, models):
    """
    Đánh giá mô hình ensemble bằng cách lấy trung bình các dự đoán.
    'models' là một danh sách các tuple (theta, beta).
    """
    total = 0
    correct = 0
    
    for i in range(len(data["user_id"])):
        u_id = data["user_id"][i]
        q_id = data["question_id"][i]
        true_label = data["is_correct"][i]
        
        predictions = []
        for theta, beta in models:
            # Đảm bảo ID nằm trong phạm vi của mảng theta/beta
            if u_id < len(theta) and q_id < len(beta):
                prob = sigmoid(theta[u_id] - beta[q_id])
                predictions.append(prob)
        
        if not predictions:
            # Trường hợp không có mô hình nào dự đoán được (hiếm)
            continue
            
        # Lấy trung bình các dự đoán (xác suất)
        avg_prob = np.mean(predictions)
        
        # Chuyển đổi xác suất trung bình thành dự đoán nhị phân
        final_guess = (avg_prob >= 0.5)
        
        if final_guess == true_label:
            correct += 1
        total += 1
        
    return correct / float(total)


def main():
    # Tải dữ liệu
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # Xác định kích thước đầy đủ (Lấy từ shape của ma trận sparse là an toàn nhất)
    # 542 students, 1774 questions
    N_USERS = 542
    N_QUESTIONS = 1774

    #####################################################################
    # TODO:                                                             #
    # 1. Chọn 3 mô hình cơ sở. (Chúng ta sẽ dùng 3 mô hình IRT)          #
    # 2. Huấn luyện từng mô hình trên một mẫu bootstrap.                 #
    # 3. Tính trung bình dự đoán và báo cáo độ chính xác.                #
    #####################################################################

    # --- Cấu hình cho mô hình cơ sở (IRT) ---
    # Bạn nên sử dụng siêu tham số (hyperparameters) tốt nhất
    # mà bạn tìm thấy khi chạy item_response.py.
    # (Giả sử đây là các giá trị tốt nhất của bạn)
    BEST_LR = 0.01
    BEST_LAMB = 0.01
    BEST_ITER = 20
    NUM_MODELS = 3

    # *** SỬA ĐỔI: Thay thế tất cả tiếng Việt có dấu bằng tiếng Anh ***
    print("Starting Ensemble (Bagging) training...")
    print(f"Number of base models: {NUM_MODELS}")
    print(f"Hyperparams (for each model): lr={BEST_LR}, lamb={BEST_LAMB}, iter={BEST_ITER}")
    print("-" * 40)

    trained_models = []

    for i in range(NUM_MODELS):
        print(f"Training model {i + 1}/{NUM_MODELS}...")
        
        # 1. Tạo mẫu bootstrap
        boot_train_data = bootstrap_data(train_data)
        
        # 2. Huấn luyện mô hình cơ sở
        theta, beta = irt(
            boot_train_data,
            val_data,
            BEST_LR,
            BEST_ITER,
            N_USERS,
            N_QUESTIONS,
            BEST_LAMB
        )
        
        # 3. Lưu mô hình đã huấn luyện
        trained_models.append((theta, beta))

    print("\nFinished training all base models.")
    print("-" * 40)

    # 4. Đánh giá mô hình Ensemble trên tập Validation và Test
    val_acc = ensemble_evaluate(val_data, trained_models)
    test_acc = ensemble_evaluate(test_data, trained_models)

    print(f"Ensemble Validation Accuracy: {val_acc:.4f}")
    print(f"Ensemble Test Accuracy: {test_acc:.4f}")
    
    print("\nCompleted Part 4 (Ensemble).")
    print("Remember: You need to compare these results with the single models")
    print("and write your explanation/discussion in the final_report.pdf.")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()