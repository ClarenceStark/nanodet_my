import os

# 定义映射关系
mapping = {0: 0, 3: 1, 9: 2, 12: 3}

# 遍历当前目录下的所有txt文件
for filename in os.listdir('.'):
    if filename.endswith('.txt'):
        # 读取文件内容
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 修改文件内容
        new_lines = []
        for line in lines:
            tokens = line.strip().split()
            if tokens:
                # 将第一个数字转换为整数并映射
                original_label = int(tokens[0])
                if original_label in mapping:
                    tokens[0] = str(mapping[original_label])
                else:
                    # 如果第一个数字不在映射中，可根据需要处理
                    pass
                new_line = ' '.join(tokens)
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        # 将修改后的内容写回文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))

