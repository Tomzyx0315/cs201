class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        d = [0]*n
        for l,r in queries:
            d[l]+=1
            d[r+1]-=1
        ans = 0
        a = [d[0]]
        if a[0]>=nums[0]:
            ans+=1
        for i in range(1,n):
            a.append(a[i-1]+d[i])
            if a[i]>=nums[i]:
                ans+=1
        if ans == n:
            return True
        return False