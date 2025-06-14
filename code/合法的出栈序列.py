x = input()
n = len(x)
while True:
    try:
        s = input()
    except EOFError:
        break
    if len(s)!=len(x):
        print('NO')
        continue
    # 不是哥们？这个特判？
    stack = []
    index1 = 0
    index2 = 0
    while index2<n and index1<n:
        stack.append(x[index1])
        index1+=1
        while stack and stack[-1]==s[index2]:
            stack.pop()
            index2+=1
    if stack:
        print('NO')
    else:
        print('YES')