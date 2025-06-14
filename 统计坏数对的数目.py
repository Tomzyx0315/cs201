class Solution:
    def countBadPairs(self, nums: List[int]) -> int:
        n = len(nums)
        number = {}
        for i in range(n):
            a = nums[i]-i
            if a not in number:
                number[a]=1
            else:
                number[a]+=1
        ans = 0
        for w in number.values():
            ans+=w*(w-1)//2
        return n*(n-1)//2-ans