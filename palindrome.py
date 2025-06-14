n = int(input())
s = input()
dp_prev = [0] * n
dp_curr = [0] * n

for i in range(n - 2, -1, -1):
    dp_curr[i] = 0  # 单字符为回文
    for j in range(i + 1, n):
        if s[i] == s[j]:
            dp_curr[j] = dp_prev[j - 1]
        else:
            dp_curr[j] = min(dp_prev[j], dp_curr[j - 1]) + 1
    dp_prev, dp_curr = dp_curr, [0] * n

print(dp_prev[n - 1])