import torch
import numpy as np
import matplotlib.pyplot as plt
from arguments import arg_parse
import ast

args = arg_parse()

def plot(singular_values):

    plt.plot(singular_values, label=f"0.{args.aug_ratio}")
    # plt.yscale('log') 
    # plt.title("Singular Value Distribution")
    # plt.xlabel("Index")
    # plt.ylabel("Singular Value (Log)")
    # plt.savefig(f"singular_value_distribution_{args.DS}.png", dpi=300, bbox_inches="tight")
    # plt.close()


if __name__ == "__main__":
    # file_list = []
    dataset_name = "MUTAG"
    for aug_ratio in range(1, 10):
        file_path = f"./logs/GCL/{dataset_name}/{dataset_name}_0.{aug_ratio}_0_singular_val"
        # file_list.append(file_path)
        with open(file_path, 'r') as f:
            singular_val = f.read()
            # singular_val = ast.literal_eval(singular_val)
            singular_val = np.array(singular_val)
            # plot(singular_val)
            plt.plot(singular_val, label=f"0.{aug_ratio}")
    plt.yscale('log') 
    plt.title("Singular Value Distribution")
    plt.xlabel("Index")
    plt.ylabel("Singular Value (Log)")
    plt.savefig(f"singular_value_distribution_{dataset_name}.png", dpi=300, bbox_inches="tight")
    plt.close()