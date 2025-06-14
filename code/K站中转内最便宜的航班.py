class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        dist = [float('inf')]*n
        dist[src]=0

        for _ in range(k + 1):  # 最多k+1条边
            new_dist = dist[:]  # 保持一份拷贝，防止使用当轮更新的结果
            for u, v, w in flights:
                if dist[u] != float('inf') and dist[u] + w < new_dist[v]:
                    new_dist[v] = dist[u] + w
            dist = new_dist

        return -1 if dist[dst] == float('inf') else dist[dst]