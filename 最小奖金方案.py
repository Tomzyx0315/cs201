'''
from functools import lru_cache
@lru_cache(maxsize=1001)
def dp(num):
    if not dic[num]:
        return 0
    MIN = float('-inf')
    for b in dic[num]:
        MIN = max(MIN,dp(b)+1)
    return MIN
n,m = map(int,input().split())
dic = {i:[] for i in range(n)}
for _ in range(m):
    a,b = map(int,input().split())
    dic[a].append(b)
w = 100*n
for i in range(n):
    w+=dp(i)
print(w)
'''
n,m = map(int,input().split())
dic = {i:[] for i in range(n)}
in_degree = [0]*n
for _ in range(m):
    a,b = map(int,input().split())
    dic[b].append(a)
    in_degree[a]+=1
from collections import deque
bonus = [100]*n
queue = deque([])
for i in range(n):
    if in_degree[i]==0:
        queue.append(i)
while queue:
    curr = queue.popleft()
    for neighbor in dic[curr]:
        bonus[neighbor] = max(bonus[neighbor],bonus[curr]+1)
        in_degree[neighbor] -= 1
        if in_degree[neighbor] == 0:
            queue.append(neighbor)

print(sum(bonus))