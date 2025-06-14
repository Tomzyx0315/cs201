s = input()
n = len(s)
ans = ''
numstack = []
stack = []
i = 0
while i < n:
    if s[i].isnumeric():
        nums = s[i]
        while i+1 < n:
            if s[i+1].isnumeric():
                i+=1
                nums+=s[i]
            else:
                numstack.append(int(nums))
                break
    elif s[i]=='[':
        stack.append(s[i])
    elif s[i]==']':
        assisted = ''
        while stack and stack[-1]!='[':
            assisted = stack.pop()+assisted
        stack.pop()
        stack.append(assisted*numstack.pop())
    else:
        stack.append(s[i])
    i+=1
#注意最外层不一定是括号！
print(''.join(stack))