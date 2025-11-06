# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, [Student Name] ([Student ID]), declare that this code is my own original work.

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
        
        # IMPORTANT: Initialize weights properly to avoid NaN
        nn.init.xavier_uniform_(self.g.weight)
        nn.init.xavier_uniform_(self.h.weight)
        nn.init.zeros_(self.g.bias)
        nn.init.zeros_(self.h.bias)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2 for regularization."""
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """
        Implement the forward pass:
        out = sigmoid(h(sigmoid(g(inputs))))
        """
        # === YOUR CODE HERE ===
        # Encoder: apply linear transformation then sigmoid
        h = torch.sigmoid(self.g(inputs))
        
        # Decoder: apply linear transformation then sigmoid
        out = torch.sigmoid(self.h(h))
        
        return out
        # === END OF YOUR CODE ===

def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, student_id=""):
    """
    Train the autoencoder with L2 regularization.
    """
    # === YOUR CODE HERE ===
    
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)  # Using SGD as in instructions
    
    # Track metrics
    train_losses = []
    val_accuracies = []
    
    num_student = train_data.shape[0]
    
    print(f"Training AutoEncoder with k={model.g.out_features}, lr={lr}, lamb={lamb}")
    print("-" * 60)
    
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0
        
        # Process each student
        for i in range(num_student):
            # Get input (zero-filled) and target (with NaN)
            inputs = Variable(zero_train_data[i]).unsqueeze(0)
            target = train_data[i]
            
            # Forward pass
            output = model(inputs)
            
            # Create mask for observed entries (not NaN)
            nan_mask = ~torch.isnan(target)
            
            # Replace NaN in target with 0 for computation
            target_clean = target.clone()
            target_clean[torch.isnan(target_clean)] = 0
            
            # Compute reconstruction loss only on observed entries
            # Loss = sum of squared errors (not averaged, as per spec)
            diff = (output.squeeze() - target_clean) ** 2
            reconstruction_loss = torch.sum(diff * nan_mask.float())
            
            # Add L2 regularization: (lamb/2) * (||W1||^2 + ||W2||^2)
            regularization_loss = (lamb / 2.0) * model.get_weight_norm()
            
            # Total loss
            loss = reconstruction_loss + regularization_loss
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"WARNING: NaN detected at epoch {epoch}, student {i}")
                print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                print(f"  Reconstruction loss: {reconstruction_loss.item()}")
                print(f"  Regularization loss: {regularization_loss.item()}")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # Average training loss
        avg_train_loss = train_loss / num_student
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set
        val_acc = evaluate(model, zero_train_data, valid_data)
        val_accuracies.append(val_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epoch} - Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    print(f"Training completed. Final Val Acc: {val_accuracies[-1]:.4f}")
    
    # Plot and save results if student_id is provided
    if student_id:
        plt.figure(figsize=(12, 5))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title(f'Training Loss (k={model.g.out_features}, λ={lamb})')
        plt.grid(True, alpha=0.3)
        
        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, linewidth=2, color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title(f'Validation Accuracy (k={model.g.out_features}, λ={lamb})')
        plt.grid(True, alpha=0.3)
        
        # Mark best accuracy
        best_acc = max(val_accuracies)
        best_epoch = val_accuracies.index(best_acc)
        plt.plot(best_epoch, best_acc, 'r*', markersize=15, 
                label=f'Best: {best_acc:.4f}')
        plt.legend()
        
        plt.tight_layout()
        filename = f'autoencoder_results_{student_id}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.close()
    
    return train_losses, val_accuracies
    # === END OF YOUR CODE ===

def evaluate(model, train_data, valid_data):
    """
    Evaluate the model on valid_data. (Already provided.)
    """
    model.eval()
    total = 0
    correct = 0
    
    with torch.no_grad():  # Important: disable gradient computation
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


    student_id = "2201010195"
    
    print("="*70)
    print("PART A.3: NEURAL NETWORK (AUTOENCODER)")
    print("="*70)
    print(f"\nData Info:")
    print(f"  Number of questions: {num_question}")
    print(f"  Number of students: {train_matrix.shape[0]}")
    print(f"  Validation set size: {len(valid_data['user_id'])}")
    
    # Hyperparameter grid - ADJUSTED for better stability
    k_values = [10, 50, 100, 200, 500]
    lr_values = [0.01, 0.001]  # Lower learning rates
    lamb_values = [0.001, 0.01, 0.1]
    num_epoch = 50
    
    # Store results
    results = {}
    best_val_acc = 0
    best_config = None
    best_model = None
    
    print("\n(c) HYPERPARAMETER TUNING")
    print("-"*70)
    print(f"Testing {len(k_values)} values of k with different lr and lambda...")
    print()
    
    # Grid search over k values
    for k in k_values:
        print(f"\n{'='*70}")
        print(f"Testing k = {k}")
        print(f"{'='*70}")
        
        # For each k, try different lr and lamb combinations
        for lr in lr_values:
            for lamb in lamb_values:
                config_name = f"k{k}_lr{lr}_lamb{lamb}"
                print(f"\nConfiguration: {config_name}")
                
                # Create and train model
                model = AutoEncoder(num_question, k)
                
                try:
                    train_losses, val_accs = train(
                        model, lr, lamb, 
                        train_matrix, zero_train_matrix, 
                        valid_data, num_epoch,
                        student_id=""  # Don't save plots for all configs
                    )
                    
                    # Store results
                    final_val_acc = val_accs[-1]
                    results[config_name] = {
                        'k': k,
                        'lr': lr,
                        'lamb': lamb,
                        'val_acc': final_val_acc,
                        'val_accs': val_accs,
                        'train_losses': train_losses
                    }
                    
                    # Track best configuration
                    if final_val_acc > best_val_acc:
                        best_val_acc = final_val_acc
                        best_config = config_name
                        best_model = model
                        print(f"  → NEW BEST: Val Acc = {final_val_acc:.4f}")
                
                except Exception as e:
                    print(f"  ERROR training this configuration: {e}")
                    continue
    
    # Report results
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*70)
    
    if len(results) == 0:
        print("ERROR: No configurations completed successfully!")
        return
    
    print(f"\n{'Config':<25} {'k':<6} {'lr':<10} {'lambda':<10} {'Val Acc':<10}")
    print("-"*70)
    
    # Sort by validation accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['val_acc'], reverse=True)
    
    for i, (config_name, result) in enumerate(sorted_results[:10]):  # Top 10
        marker = "★" if i == 0 else " "
        print(f"{marker} {config_name:<23} {result['k']:<6} {result['lr']:<10.4f} "
              f"{result['lamb']:<10.4f} {result['val_acc']:<10.4f}")
    
    print("\n" + "="*70)
    print("BEST CONFIGURATION")
    print("="*70)
    best_result = results[best_config]
    print(f"  Configuration: {best_config}")
    print(f"  k (latent dimension): {best_result['k']}")
    print(f"  Learning rate: {best_result['lr']}")
    print(f"  Regularization lambda: {best_result['lamb']}")
    print(f"  Validation Accuracy: {best_val_acc:.4f}")
    
    # (d) Result Analysis - Retrain best model with saved plot
    print("\n" + "="*70)
    print("(d) RESULT ANALYSIS - Retraining Best Model")
    print("="*70)
    
    final_model = AutoEncoder(num_question, best_result['k'])
    final_train_losses, final_val_accs = train(
        final_model,
        best_result['lr'],
        best_result['lamb'],
        train_matrix,
        zero_train_matrix,
        valid_data,
        num_epoch,
        student_id=student_id  # Save plot for best config
    )
    
    # (e) Test Accuracy and Model Selection
    print("\n" + "="*70)
    print("(e) TEST SET EVALUATION")
    print("="*70)
    
    test_acc = evaluate(final_model, zero_train_matrix, test_data)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_accs[-1]:.4f}")
    
    # Compare across k values (using best lr and lamb)
    print(f"\nComparison across k values (lr={best_result['lr']}, lamb={best_result['lamb']}):")
    print(f"{'k':<10} {'Val Acc':<15} {'Test Acc':<15}")
    print("-"*50)
    
    for k in k_values:
        config = f"k{k}_lr{best_result['lr']}_lamb{best_result['lamb']}"
        if config in results:
            val_acc = results[config]['val_acc']
            
            # Quick test evaluation
            model_k = AutoEncoder(num_question, k)
            train(model_k, best_result['lr'], best_result['lamb'],
                 train_matrix, zero_train_matrix, valid_data, 30, "")
            test_acc_k = evaluate(model_k, zero_train_matrix, test_data)
            
            marker = "★" if k == best_result['k'] else " "
            print(f"{marker} {k:<8} {val_acc:<15.4f} {test_acc_k:<15.4f}")
    
    # (f) Reflection
    print("\n" + "="*70)
    print("(f) REFLECTION")
    print("="*70)
    
    print("\n1. Effect of Regularization (λ):")
    print("   • Larger λ reduces overfitting by penalizing large weights")
    print("   • Too large λ can cause underfitting and limit model expressiveness")
    print("   • Optimal λ balances model complexity and generalization")
    print(f"   • Best λ found: {best_result['lamb']}")
    
    print("\n2. Effect of Latent Dimension (k):")
    print("   • Small k (e.g., 10): May underfit, cannot capture complex student-question patterns")
    print("   • Large k (e.g., 500): More expressive but risks overfitting to training noise")
    print("   • Medium k provides good trade-off between representation power and generalization")
    print(f"   • Best k found: {best_result['k']}")
    
    print("\n3. Observations:")
    print("   • Training loss generally decreases over epochs showing learning")
    print("   • Validation accuracy may plateau, indicating convergence")
    print("   • Neural networks provide flexible non-linear modeling of student abilities")
    print("   • Proper initialization and gradient clipping help training stability")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

if __name__ == "__main__":
    main()