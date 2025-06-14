class Solution:
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        from collections import deque,defaultdict
        queue = deque([])
        topology = []
        n = len(colors)
        in_degree = [0]*n
        graph = defaultdict(list)
        for a,b in edges:
            in_degree[b]+=1
            graph[a].append(b)
        for i in range(n):
            if in_degree[i]==0:
                queue.append(i)
        while queue:
            point = queue.popleft()
            topology.append(point)
            if point in graph:
                for nextpoint in graph[point]:
                    in_degree[nextpoint]-=1
                    if in_degree[nextpoint]==0:
                        queue.append(nextpoint)
        if len(topology)!=n:
            return -1
        max_number = [[0]*26 for _ in range(n)]
        for point in topology:
            color_idx = ord(colors[point]) - ord('a')
            max_number[point][color_idx] += 1
            for v in graph[point]:
                for c in range(26):
                    max_number[v][c] = max(max_number[v][c], max_number[point][c])
        return max([max(max_number[point]) for point in range(n)])