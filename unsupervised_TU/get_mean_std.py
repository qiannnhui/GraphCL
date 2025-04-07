import numpy as np

# 假設你的檔案名是文件夾中的 txt 檔案
DS = "REDDIT-BINARY"
# mode = "pull_negative_rm_FN"
# mode = "rm_FN"
# mode = "cheated"
mode = "pull_negative"
# mode = "single_other_pos"
aug_ratio = 0.1
file_path = f"logs/old/{mode}/{DS}"
file_paths = []
for i in range(5):
    file_paths.append(f"{file_path}/{DS}_{aug_ratio}_{i}")

max_values = []

for file_path in file_paths:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # 假設 test 的數據在最後一行，並且是這樣的格式: {"val": [...], "test": [...] }
        for line in lines:
            if '"test"' in line:
                # 提取 "test" 的數據
                start_index = line.find('"test":') + len('"test":') + 1
                end_index = line.find(']', start_index) + 1
                test_values_str = line[start_index:end_index]
                test_values = eval(test_values_str)  # 把字串轉換為列表
                
                # 取得 "test" 部分的最大值
                max_values.append(max(test_values))

# 計算平均值和標準差
mean_max_value = np.mean(max_values)
std_max_value = np.std(max_values)

print(f"best mean ({DS}_{mode}): {mean_max_value} ± {std_max_value}")
# print(f"best std ({DS}_{mode}): {std_max_value}")
