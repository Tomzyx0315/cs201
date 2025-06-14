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
while True:
    try:
        n = input()
    except EOFError:
        break
    n = int(n)
    lines = []
    for i in range(n):
        a = list(map(int,input().split()))
        for j in range(i+1,n):
            lines.append((a[j],i,j))
        
    lines.sort()
    disjoint = DisjointSet(n)
    cnt = 0
    tot = 0
    for weight, start, final in lines:
        if cnt == n-1:
            print(tot)
            break
        if disjoint.find(start)!=disjoint.find(final):
            cnt+=1
            tot+=weight
            disjoint.union(start,final)
    else:
        print(tot)