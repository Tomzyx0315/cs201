class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans = []
        n = len(nums)
        curr = []
        def dfs(i):
            if i==n:
                ans.append(curr[::])
                return
            dfs(i+1)
            curr.append(nums[i])
            dfs(i+1)
            curr.pop()
        dfs(0)
        return ans