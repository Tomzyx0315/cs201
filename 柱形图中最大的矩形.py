from typing import List
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        heights = [0]+heights+[0]
        rec = 0
        for i in range(len(heights)):
            while stack and heights[stack[-1]]>heights[i]:
                height = heights[stack.pop()]
                w = i-stack[-1]-1
                rec = max(rec,height*w)
            stack.append(i)
        return rec