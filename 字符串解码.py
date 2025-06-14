class Solution:
    def decodeString(self, s: str) -> str:
        n = len(s)
        stack = []
        nums = []
        i = 0
        while i<len(s):
            if s[i].isnumeric():
                num = s[i]
                i+=1
                while i<len(s):
                    if s[i].isnumeric():
                        num+=s[i]
                        i+=1
                    else:
                        break
                nums.append(num)
                continue
            elif s[i]=='[':
                stack.append(s[i])
            elif s[i]==']':
                assisted = ''
                while stack and stack[-1]!='[':
                    assisted = stack.pop()+assisted
                stack.pop()
                stack.append(int(nums.pop())*assisted)
            else:
                stack.append(s[i])
            i+=1
        return ''.join(stack)