import random
import string

# 生成 500 个随机名字
names = []
for _ in range(500):
    length = random.randint(3, 8)
    name = ''.join(random.choices(string.ascii_lowercase, k=length)).capitalize()
    names.append(name)

with open('input.txt', 'w') as f:
    for name in names:
        f.write(name + '\n')
print("500个随机名字已写入 input.txt！")