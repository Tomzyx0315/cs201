import heapq
def minroute(start,final):
    if start == final:
        print(start)
        return
    distances = {}
    backtrack = {}
    for loc in mapping:
        distances[loc] = float('inf')
    distances[start] = 0
    priority_queue = [(0,start)]
    while priority_queue:
        distance, x = heapq.heappop(priority_queue)
        if distance > distances[x]:
            continue
        if x==final:
            break
        for nextmove,d in mapping[x].items():
            newdistance = distance+d
            if newdistance<distances[nextmove]:
                distances[nextmove]=newdistance
                backtrack[nextmove]=x
                heapq.heappush(priority_queue,(newdistance,nextmove))
    route = final
    nextpoint = final
    while True:
        lastpoint = backtrack[nextpoint]
        route = lastpoint+f'->({mapping[lastpoint][nextpoint]})->'+route
        nextpoint = lastpoint
        if lastpoint==start:
            break
    print(route)
p = int(input())
mapping = {input():{} for _ in range(p)}
q = int(input())
for _ in range(q):
    a,b,c = input().split()
    c = int(c)
    mapping[a][b]=c
    mapping[b][a]=c
r = int(input())
for _ in range(r):
    start,final = input().split()
    minroute(start,final)