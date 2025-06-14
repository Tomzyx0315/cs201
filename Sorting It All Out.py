from collections import deque
while True:
    n,m = map(int,input().split())
    if n==m==0:
        break
    relation = []
    for _ in range(m):
        a,b = input().split('<')
        relation.append((b,a))
    for i in range(1,m+1):
        graph = {chr(ord('A')+i):[] for i in range(n)}
        in_degree = {chr(ord('A')+i):0 for i in range(n)}
        for j in range(i):
            b,a = relation[j]
            in_degree[a]+=1
            graph[b].append(a)
        queue = deque([])
        topology = []
        prelist = [chr(ord('A')+i) for i in range(n)]
        cnt = 0
        for word in prelist:
            if in_degree[word]==0:
                queue.append(word)
        while queue:
            if len(queue)!=1:
                cnt+=1
            point = queue.popleft()
            topology.append(point)
            if point in graph:
                for nextpoint in graph[point]:
                    in_degree[nextpoint]-=1
                    if in_degree[nextpoint]==0:
                        queue.append(nextpoint)
        if n!=len(topology):
            print(f'Inconsistency found after {i} relations.')
            break
        if n==len(topology) and not cnt:
            ans = ''.join(topology[::-1])
            print(f'Sorted sequence determined after {i} relations: {ans}.')
            break
    else:
        print('Sorted sequence cannot be determined.')