n,m = map(int,input().split())
class Vertex:
    def __init__(self):
        self.connection = set()
        self.degree = 0
    def connect(self,another):
        self.connection.add(another)
        self.degree+=1
class Graph:
    def __init__(self,n):
        self.graph = [Vertex() for _ in range(n)]
    def display(self,n):
        for i in range(n):
            ans = []
            for j in range(n):
                if i==j:
                    ans.append(str(self.graph[i].degree))
                else:
                    ans.append('-1' if self.graph[j] in self.graph[i].connection else '0')
            print(' '.join(ans))
    def connect(self,i,j):
        self.graph[i].connect(self.graph[j])
        self.graph[j].connect(self.graph[i])
graph = Graph(n)
for _ in range(m):
    a,b = map(int,input().split())
    graph.connect(a,b)
graph.display(n)