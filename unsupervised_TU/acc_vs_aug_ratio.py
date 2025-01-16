import matplotlib.pyplot as plt

# Data for plotting
aug_ratio = data["aug_ratio"]
mutag_mean = data["MUTAG_best_mean"]
mutag_std = data["MUTAG_best_std"]
proteins_mean = data["PROTEINS_mean"]
proteins_std = data["PROTEINS_std"]

# Plot
plt.figure(figsize=(10, 6))

# MUTAG line
plt.errorbar(aug_ratio, mutag_mean, yerr=mutag_std, fmt='-o', capsize=5, label="MUTAG", color="blue")

# PROTEINS line (ignore None values)
valid_proteins_indices = [i for i, v in enumerate(proteins_mean) if v is not None]
valid_proteins_mean = [proteins_mean[i] for i in valid_proteins_indices]
valid_proteins_std = [proteins_std[i] for i in valid_proteins_indices]
valid_aug_ratio = [aug_ratio[i] for i in valid_proteins_indices]

plt.errorbar(valid_aug_ratio, valid_proteins_mean, yerr=valid_proteins_std, fmt='-o', capsize=5, label="PROTEINS", color="green")

# Add labels, title, and legend
plt.title("Accuracy vs Augmentation Ratio", fontsize=16)
plt.xlabel("Augmentation Ratio", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()
