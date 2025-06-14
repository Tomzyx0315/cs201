import heapq
class Solution:
    def findMaxSum(self, nums1: list[int], nums2: list[int], k: int) -> list[int]:
        n = len(nums1)
        ans = [0]*n
        wrapup = list(enumerate(nums1))
        wrapup.sort(key=lambda x:x[1])
        minheap = []
        heaplen = 0
        heapsum = 0
        for i in range(n):
            if i==0:
                continue
            if wrapup[i][1]==wrapup[i-1][1]:
                ans[wrapup[i][0]]=ans[wrapup[i-1][0]]
                if heaplen<k:
                    heaplen+=1
                    heapq.heappush(minheap,nums2[wrapup[i-1][0]])
                    heapsum+=nums2[wrapup[i-1][0]]
                else:
                    popout = heapq.heappop(minheap)
                    heapsum-=popout
                    heapq.heappush(minheap,max(nums2[wrapup[i-1][0]],popout))
                    heapsum+=max(nums2[wrapup[i-1][0]],popout)
                continue
            else:
                if heaplen<k:
                    heaplen+=1
                    heapq.heappush(minheap,nums2[wrapup[i-1][0]])
                    heapsum+=nums2[wrapup[i-1][0]]
                else:
                    popout = heapq.heappop(minheap)
                    heapsum-=popout
                    heapq.heappush(minheap,max(nums2[wrapup[i-1][0]],popout))
                    heapsum+=max(nums2[wrapup[i-1][0]],popout)
                ans[wrapup[i][0]]=heapsum
        return ans