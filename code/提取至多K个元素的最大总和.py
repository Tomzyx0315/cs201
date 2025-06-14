from typing import List
class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        n = len(grid)
        m = len(grid[0])
        ans = []
        for i in range(n):
            nums = grid[i]
            nums = sorted(nums,reverse=True)
            for j in range(limits[i]):
                if nums[j]<=0:
                    break
                ans.append(nums[j])
        ans = sorted(ans,reverse=True)
        if len(ans)<k:
            return sum(ans)
        return sum(ans[:k])