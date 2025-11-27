import matplotlib.pyplot as plt
import numpy as np
import os

def plot_theta_per_epoch(args, all_epoch_thetas, save_dir=None):
    if save_dir is None:
        save_dir = f'./result/{args.DS}/reweighted_{args.reweighted_loss}/theta_loss_{args.theta_loss}/dist_{args.dist}/angles'
    os.makedirs(save_dir, exist_ok=True)

    # epochs = np.arange(len(all_epoch_thetas))
    epochs = np.arange(0, len(all_epoch_thetas) * args.log_interval, args.log_interval)

    # --- Q1-Q3 ---
    q1_list, q2_list, q3_list, mean_list = [], [], [], []

    for arr in all_epoch_thetas:
        if len(arr) > 0:
            q1 = np.percentile(arr, 25)
            q2 = np.percentile(arr, 50)
            q3 = np.percentile(arr, 75)
            mean = np.mean(arr)
        else:
            q1, q2, q3, mean = 0, 0, 0, 0
        q1_list.append(q1)
        q2_list.append(q2)
        q3_list.append(q3)
        mean_list.append(mean)

    plt.figure(figsize=(10,6))

    # scatter plot of all points
    for i, arr in enumerate(all_epoch_thetas):
        if len(arr) > 0:
            epoch_val = epochs[i]  # 真實 epoch
            jitter = np.random.uniform(-0.3, 0.3, size=arr.shape)
            plt.scatter(np.full_like(arr, epoch_val) + jitter, arr, alpha=0.1, s=6, color='gray')

    # plot Q1 and Q3 shaded area
    plt.fill_between(epochs, q1_list, q3_list, color='skyblue', alpha=0.3, label='IQR (Q1–Q3)')
    # Q2 line
    plt.plot(epochs, q2_list, color='blue', linewidth=2, label='Median (Q2)')
    # Mean line
    plt.plot(epochs, mean_list, color='orange', linestyle='--', linewidth=2, label='Mean')

    plt.xlabel('Epoch')
    plt.ylabel('Angle (degree)')
    plt.title('Angle distribution across epochs (Scatter + Quartiles)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'angle_distribution_scatter_quartiles.png'), dpi=300)
    plt.close()
