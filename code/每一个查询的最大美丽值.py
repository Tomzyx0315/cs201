from typing import List
from bisect import bisect_right
class Solution:
    def maximumBeauty(self, items: List[List[int]], queries: List[int]) -> List[int]:
        ans = []
        n = len(items)
        items.sort(key=lambda x:x[0])
        prelist = [0]*(n+1)
        for i in range(1,n+1):
            prelist[i] = max(prelist[i-1],items[i-1][1])
        items = [i[0] for i in items]
        for q in queries:
            #index = bisect_right(items,(q,0))# 这个排序第二个元素也会被涉及到，所以得改
            index = bisect_right(items,q)
            ans.append(prelist[index])
        return ans