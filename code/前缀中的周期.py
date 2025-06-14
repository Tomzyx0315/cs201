cnt = 0
while True:
    n = int(input())
    if n == 0:
        break
    cnt += 1
    if cnt != 1:
        print()
    print(f'Test case #{cnt}')
    s = input()
    
    pi = [0] * n  # 前缀函数
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j

    for i in range(n):
        length = i + 1
        k = pi[i]
        if k > 0 and length % (length - k) == 0:
            print(length, length // (length - k))
'''
构建前缀函数数组 pi，其中 pi[i] 表示字符串前缀 s[0..i] 中最长的相等前后缀的长度。

之后对于每个前缀 s[0..i]，设其长度为 L = i + 1，若它的最长前后缀长度为 k = pi[i]，并且满足 L % (L - k) == 0，就说明该前缀是由某个更短的子串重复构成的，重复次数为 L / (L - k)。
'''