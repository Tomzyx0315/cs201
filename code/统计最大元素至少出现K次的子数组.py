class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans = 0
        MAX = max(nums)
        left = 0
        cur = 0
        for right in range(len(nums)):
            if nums[right]==MAX:
                cur+=1
            while cur>=k:
                ans+=(len(nums)-right)
                if nums[left]==MAX:
                    cur-=1
                left+=1

        return ans