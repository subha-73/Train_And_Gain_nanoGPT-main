import re
import matplotlib.pyplot as plt

log_file = 'training_log.txt'

train_iters = []
train_losses = []
val_iters = []
val_losses = []

with open(log_file, 'r') as f:
    for line in f:
        # Match lines like: iter 0: loss 4.2886
        train_match = re.search(r'iter (\d+): loss ([\d.]+)', line)
        if train_match:
            train_iters.append(int(train_match.group(1)))
            train_losses.append(float(train_match.group(2)))

        # Match lines like: step 500: train loss 2.34, val loss 2.41
        val_match = re.search(r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)', line)
        if val_match:
            val_iters.append(int(val_match.group(1)))
            val_losses.append(float(val_match.group(3)))

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(train_iters, train_losses, label='Training Loss', color='green', alpha=0.6)
plt.plot(val_iters, val_losses, label='Validation Loss', color='red', marker='o')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Iterations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()