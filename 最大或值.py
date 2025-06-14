#枚举，前后缀和
class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # suf[i]表示nums[i+1:]的OR值
        suf = [0]*n
        for i in range(n-2,-1,-1):
            suf[i]=suf[i+1] | nums[i+1]
        # pre表示nums[:i]的OR值
        ans = pre = 0
        for i in range(n):
            ans = max(ans,pre | (nums[i]<<k) | suf[i])
            pre |= nums[i]
        return ans