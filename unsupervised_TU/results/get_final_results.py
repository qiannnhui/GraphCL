import re
import numpy as np

def get_one_result(DS):
    file_dir = f"/home/qiannnhui/qiannnhui/GraphCL/unsupervised_TU/results/GCL/{DS}/reweighted_InfoNCE_GCL_new.log"
    # /home/qiannnhui/qiannnhui/GraphCL/unsupervised_TU/results/GCL/COLLAB/reweighted_InfoNCE_GCL_new.log
    with open(file_dir, 'r') as file:
        log_text = file.read()

    all_best_val_acc = [float(match) for match in re.findall(r"Best Val Accuracy:\s+([0-9.]+)", log_text)]

    best_val_acc = all_best_val_acc[-5:]

    mean_val = np.mean(best_val_acc)
    std_val = np.std(best_val_acc)

    print("Best Val Accuracies:", best_val_acc)
    print("Mean:", round(mean_val, 6))
    print("Standard Deviation:", round(std_val, 6))
    print(f"{DS} Final Results:", round(mean_val*100, 2), "±", round(std_val*100, 2), "\n")


if __name__ == "__main__":
    for DS in ['NCI1', 'PROTEINS', 'DD', 'MUTAG', 'COLLAB', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'IMDB-BINARY']:
        get_one_result(DS)
# file_dir = f"/home/qiannnhui/GeoGCL/unsupervised_TU/result/{DS}/log_geogcl_distance_2.0_angle_180.0.log"
# with open(file_dir, 'r') as file:
#     log_text = file.read()

# all_best_val_acc = [float(match) for match in re.findall(r"Best Val Accuracy:\s+([0-9.]+)", log_text)]

# best_val_acc = all_best_val_acc[-5:]

# mean_val = np.mean(best_val_acc)
# std_val = np.std(best_val_acc)

# print("Best Val Accuracies:", best_val_acc)
# print("Mean:", round(mean_val, 6))
# print("Standard Deviation:", round(std_val, 6))
# print(f"{DS} Final Results:", round(mean_val*100, 2), "±", round(std_val, 3))
