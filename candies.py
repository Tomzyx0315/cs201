# 注意这个difference指的是flymouse-snoopy没有绝对值！否则做不了了
import heapq
inf = float('inf')
n,m = map(int,input().split())
graph = [[] for _ in range(n)]# 不要用字典！要考虑到可能有重边
for _ in range(m):
    a,b,c=map(int,input().split())
    graph[a-1].append((b-1,c))
distance = [inf]*n
distance[0]=0
priority_queue = [(0,0)]
while priority_queue:
    curd,cur = heapq.heappop(priority_queue)
    if curd>distance[cur]:# 剪枝
        continue
    if cur == n-1:
        print(curd)
        exit()# 提前退出
    for next,d in graph[cur]:
        newd = curd+d
        if newd < distance[next]:
            distance[next] = newd
            heapq.heappush(priority_queue,(newd,next))
print(distance[n-1])