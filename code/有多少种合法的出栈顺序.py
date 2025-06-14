n = int(input())
catalan = [0] * (n + 1)
catalan[0] = 1  # C0 = 1

for i in range(1, n + 1):
    for j in range(i):
        catalan[i] += catalan[j] * catalan[i - 1 - j]

print(catalan[n])

# 栈深度不为负数，理解为括号匹配问题容易想明白递推公式