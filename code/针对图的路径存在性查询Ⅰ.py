from bisect import bisect_right
class Solution:
    def pathExistenceQueries(self, n: int, nums: List[int], maxDiff: int, queries: List[List[int]]) -> List[bool]:
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

        DJS = DisjointSet(n)
        m=0
        for i in range(n):
            b=bisect_right(nums,nums[i]+maxDiff)-1
            for j in range(max(m,i),b+1):
                DJS.union(i,j)
            m=max(m,b)
        # 神奇的优化！！
        ans = []
        for i in range(len(queries)):
            if DJS.find(queries[i][0])==DJS.find(queries[i][1]):
                ans.append(True)
            else:
                ans.append(False)
        return ans