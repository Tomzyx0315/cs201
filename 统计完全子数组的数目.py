class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        cnt = 0
        set1 = set()
        for num in nums:
            set1.add(num)
        n = len(set1)
        words = {}
        wordsnum = 0
        left = 0
        for right in range(len(nums)):
            if nums[right] in words:
                words[nums[right]]+=1
            else:
                words[nums[right]]=1
                wordsnum+=1
            while wordsnum==n:
                cnt+=len(nums)-right
                words[nums[left]]-=1
                if words[nums[left]]==0:
                    words.pop(nums[left])
                    wordsnum-=1
                left+=1
        return cnt