"""Generate V2 training loss figure for IEEE paper."""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load V2 training history
with open('/Users/I558118/Documents/Projects/GenreMaster/results/logs/training_history_v2.json', 'r') as f:
    history = json.load(f)

epochs = [e['epoch'] for e in history['epochs']]
train_loss = [e['train_loss'] for e in history['epochs']]
val_loss = [e['val_loss'] for e in history['epochs']]

# Find best epoch
best_epoch = min(range(len(val_loss)), key=lambda i: val_loss[i]) + 1
best_val_loss = min(val_loss)

# Create figure
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='Training Loss', linewidth=2, color='#2E86AB')
plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='#A23B72')

# Mark best epoch
plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
plt.plot(best_epoch, best_val_loss, 'r*', markersize=15, label=f'Best Val Loss: {best_val_loss:.4f} @ Epoch {best_epoch}')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('GenreMaster Training Trajectory', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save figure
plt.savefig('IEEE_PAPER/figures/fig_transformer_training_loss.png', dpi=300, bbox_inches='tight')
print(f"Figure saved to IEEE_PAPER/figures/fig_transformer_training_loss.png")
print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
