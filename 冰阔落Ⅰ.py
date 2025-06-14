class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        
        if rootX != rootY:
            print('No')
            self.parent[rootY] = rootX
        else:
            print('Yes')
while True:
    try:
        n,m = map(int,input().split())
        djs = DisjointSet(n)
        for _ in range(m):
            x,y = map(int,input().split())
            x,y = x-1,y-1
            djs.union(x,y)
        num = []
        for i in range(n):
            if djs.parent[i]==i:
                num.append(i+1)
        print(len(num))
        print(' '.join(list(map(str,num))))
    except EOFError:
        break