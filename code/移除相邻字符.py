class Solution:
    def resultingString(self, s: str) -> str:
        stack = []
        for i in s:
            if not stack:
                stack.append(i)
            else:
                if abs(ord(i)-ord(stack[-1])) in [1,25]:
                    stack.pop()
                else:
                    stack.append(i)
        return ''.join(stack)