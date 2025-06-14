class Solution:
    def trap(self, height: List[int]) -> int:
        tot = 0
        stack = []
        n = len(height)
        i = 0
        while i<n:
            while stack and height[stack[-1]]<height[i]:
                index = stack.pop()
                h = height[index]
                if stack:
                    tot += (min(height[stack[-1]],height[i])-h)*(i-stack[-1]-1)
            stack.append(i)
            i+=1
        return tot