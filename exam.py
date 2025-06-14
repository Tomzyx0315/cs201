n = int(input())
cnt = 0
stack = []
for _ in range(2*n):
    s = input().split()
    if len(s)!=1:
        stack.append(int(s[1]))
    else:
        if stack[-1]!=min(stack):
            stack.sort(reverse=True)
            cnt+=1
        stack.pop()
print(cnt)