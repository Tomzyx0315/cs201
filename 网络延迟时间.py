import heapq
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        distances = [float('inf')]*n
        distances[k-1] = 0
        priority_queue = [(0,k)]
        mapping = {}
        for u,v,w in times:
            if u not in mapping:
                mapping[u]=[(v,w)]
            else:
                mapping[u].append((v,w))
        while priority_queue:
            distance, x = heapq.heappop(priority_queue)
            if x in mapping:
                for nextmove,d in mapping[x]:
                    newdistance = distance+d
                    if newdistance<distances[nextmove-1]:
                        distances[nextmove-1]=newdistance
                        heapq.heappush(priority_queue,(newdistance,nextmove))
        MAX = max(distances)
        if MAX==float('inf'):
            return -1
        return MAX