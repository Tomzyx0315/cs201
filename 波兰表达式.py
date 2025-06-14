s = input().split()[::-1]
stack = []
for i in s:
    if i in ['+','-','*','/']:
        a = stack.pop()
        b = stack.pop()
        stack.append(str(eval(a+i+b)))
    else:
        stack.append(i)
print(f'{float(stack[-1]):.6f}')