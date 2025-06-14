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
n = int(input())
lines = []
for _ in range(n-1):
    a = input().split()
    start = ord(a[0])-ord('A')
    if len(a)!=2:
        for i in range(2,len(a),2):
            final = ord(a[i])-ord('A')
            weight = int(a[i+1])
            lines.append((weight,start,final))
        
lines.sort()
disjoint = DisjointSet(n)
cnt = 0
tot = 0
for weight, start, final in lines:
    if cnt == n-1:
        print(tot)
        exit()
    if disjoint.find(start)!=disjoint.find(final):
        cnt+=1
        tot+=weight
        disjoint.union(start,final)
print(tot)