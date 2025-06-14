class DisjointSet:
    def __init__(self,n):
        self.parent = list(range(n))
    def find(self,x):
        if self.parent[x]!=x:
            self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,x,y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX!=rootY:
            self.parent[rootY] = rootX

n = int(input())
djs = DisjointSet(26)
flag = True
equal = []
unequal = []
for _ in range(n):
    s = input()
    a,b = s[0],s[-1]
    a = ord(a)-ord('a')
    b = ord(b)-ord('a')
    if s[1]=='=':
        equal.append((a,b))
    else:
        unequal.append((a,b))
for a,b in equal:
    djs.union(a,b)
for a,b in unequal:
    if djs.find(a)==djs.find(b):
        print(False)
        exit()
print(True)