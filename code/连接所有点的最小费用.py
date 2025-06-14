class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        lines = []
        n = len(points)
        disjoint = DisjointSet(n)
        for i in range(n):
            for j in range(i+1,n):
                x1,y1 = points[i]
                x2,y2 = points[j]
                lines.append((abs(x2-x1)+abs(y2-y1),i,j))
        lines.sort()
        cnt = 0
        tot = 0
        for weight, start, final in lines:
            if cnt == n-1:
                return tot
            if disjoint.find(start)!=disjoint.find(final):
                cnt+=1
                tot+=weight
                disjoint.union(start,final)
        return tot