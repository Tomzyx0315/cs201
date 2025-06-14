class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        from collections import deque
        graph = {i:[] for i in range(numCourses)}
        in_degree = [0]*numCourses
        for a,b in prerequisites:
            graph[b].append(a)
            in_degree[a]+=1
        queue = deque([])
        topology = []
        for i in range(numCourses):
            if in_degree[i]==0:
                queue.append(i)
        while queue:
            point = queue.popleft()
            topology.append(point)
            for nextpoint in graph[point]:
                in_degree[nextpoint]-=1
                if in_degree[nextpoint]==0:
                    queue.append(nextpoint)
        if len(topology)==numCourses:
            return topology
        else:
            return []