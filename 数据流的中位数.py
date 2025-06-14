from heapq import *
# 注意python的堆是最小堆，想要用最大堆记得取负！很难debug！
class MedianFinder:

    def __init__(self):
        self.minheap = []
        self.maxheap = []

    def addNum(self, num: int) -> None:
        heappush(self.maxheap, -num)
        
        # 确保最大堆堆顶 <= 最小堆堆顶
        if self.minheap and -self.maxheap[0] > self.minheap[0]:
            heappush(self.minheap, -heappop(self.maxheap))

        # 保证两个堆大小平衡，最多相差1
        if len(self.maxheap) > len(self.minheap) + 1:
            heappush(self.minheap, -heappop(self.maxheap))
        elif len(self.minheap) > len(self.maxheap):
            heappush(self.maxheap, -heappop(self.minheap))

    def findMedian(self) -> float:
        if len(self.maxheap)>len(self.minheap):
            return -self.maxheap[0]
        else:
            return (-self.maxheap[0]+self.minheap[0])/2


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()