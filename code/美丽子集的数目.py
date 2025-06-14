from typing import List
from collections import defaultdict
class Solution:
    def beautifulSubsets(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        self.cnt = 0
        cannot = defaultdict(int)
        def dfs(i):
            if i==n:
                self.cnt+=1
                return
            num = nums[i]
            if num not in cannot:
                dfs(i+1)
                cannot[num+k]+=1
                dfs(i+1)
                cannot[num+k]-=1
                if cannot[num+k]==0:
                    del cannot[num+k] # defaultdict 容易忘记删除！
            else:
                dfs(i+1)
        dfs(0)
        return self.cnt-1