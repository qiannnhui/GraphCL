import os
import statistics
import numpy as np
import matplotlib.pyplot as plt

plot = False
dataset_name = "MUTAG"
# node, edge, attr
SADA_aug = "attr"
aug_ratio = 1
file_list = []
for i in range(5):
    # file_path = f"./logs/GCL/{dataset_name}/{dataset_name}_0.{aug_ratio}_OR_{i}"
    file_path = f"./logs/GCL/{dataset_name}/{dataset_name}_0.{aug_ratio}_{i}"
    file_list.append(file_path)


# file_list = [file1_path, file2_path, file3_path, file4_path, file5_path]

last_epoch = 20

SVC = []

for file_path in file_list:
    # print("filename = ", file_path)
    with open(file_path, 'r') as f:
        log = f.read()
        
        new_model_epoch_start_pos = log.split('[')[1]
        new_model_epoch_start = int(new_model_epoch_start_pos.split(']')[0])
        # print(new_model_epoch_start)
        
        accs = log.split('{')[1]
        #logreg_accs, svc_accs, linearsvc_accs, forest_accs = accs.split('[')[1], accs.split('[')[2], accs.split('[')[3], accs.split('[')[4]
        SVC_accs = accs.split('[')[2]
        
        tmp_list = []
        for i, acc in enumerate(SVC_accs.split(',')):
            if i < (new_model_epoch_start):
                pass
            if i == (last_epoch+new_model_epoch_start):
                break
            if i == (len(SVC_accs.split(','))-1):
                acc = float(acc[:-4])
                tmp_list.append(acc)
            else:
                acc = float(acc)
                tmp_list.append(acc)
        SVC.append(tmp_list)

list1 = SVC[0]
list2 = SVC[1]
list3 = SVC[2]
list4 = SVC[3]
list5 = SVC[4]

if plot:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(list1)+1), list1, label='0')
    plt.plot(range(1, len(list2)+1), list2, label='1')
    plt.plot(range(1, len(list3)+1), list3, label='2')
    plt.plot(range(1, len(list4)+1), list4, label='3')
    plt.plot(range(1, len(list5)+1), list5, label='4')

    plt.title('Training Curve Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accs')
    plt.legend()

    # plt.savefig(f"./logs/plots/{dataset_name}/{dataset_name}_Training_Curve_Comparison_OR_0.{aug_ratio}.png")
    plt.savefig(f"./logs/plots/{dataset_name}_old/{dataset_name}_Training_Curve_Comparison_0.{aug_ratio}.png")

print('=====EXP0=====')
print(f'Accuracy mean={statistics.mean(list1)}, max={max(list1)}, min={min(list1)}')
print('=====EXP1=====')
print(f'Accuracy mean={statistics.mean(list2)}, max={max(list2)}, min={min(list2)}')
print('=====EXP2=====')
print(f'Accuracy mean={statistics.mean(list3)}, max={max(list3)}, min={min(list3)}')
print('=====EXP3=====')
print(f'Accuracy mean={statistics.mean(list4)}, max={max(list4)}, min={min(list4)}')
print('=====EXP4=====')
print(f'Accuracy mean={statistics.mean(list5)}, max={max(list5)}, min={min(list5)}')
acc_list = [max(list1), max(list2), max(list3), max(list4), max(list5)]
print('=====Final=====')
print(f'Mean/standard deviation of accuracy = {statistics.mean(acc_list)} ± {statistics.pstdev(acc_list)}')