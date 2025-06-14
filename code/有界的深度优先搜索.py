n,m,l = map(int,input().split())
graph = {i:[] for i in range(n)}
for _ in range(m):
    a,b = map(int,input().split())
    graph[a].append(b)
    graph[b].append(a)
start = int(input())
visited = [0] * n
reached = [start]
visited[start]=1
for i in range(n):
    graph[i].sort()
def dfs(x,k):
    if k==l:
        return
    for point in graph[x]:
        if not visited[point]:
            visited[point]=1
            if point not in reached:
                reached.append(point)
            dfs(point,k+1)
            #visited[point]=0
dfs(start,0)
print(*reached)