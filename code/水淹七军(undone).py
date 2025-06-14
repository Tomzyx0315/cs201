from collections import deque
steps = [(0,1),(0,-1),(1,0),(-1,0)]
for _ in range(int(input())):
    flag = False
    m,n = map(int,input().split())
    height = [list(map(int,input().split())) for __ in range(m)]
    i,j = map(int,input().split())
    i-=1
    j-=1
    for __ in range(int(input())):
        x,y = map(int,input().split())
        x-=1
        y-=1
        if x==i and y==j:
            flag=True
        h0 = height[x][y]
        queue = deque([(x,y)])
        in_queue = set((x,y))
        if flag:
            continue
        while queue and not flag:
            fx,fy = queue.popleft()
            for dx,dy in steps:
                nx,ny = fx+dx,fy+dy
                if 0<=nx<m and 0<=ny<n and (nx,ny) not in in_queue and height[nx][ny]<=h0:
                    queue.append((nx,ny))
                    in_queue.add((nx,ny))
                    if nx==i and ny==j:
                        if height[nx][ny]<h0:
                            flag = True
    if flag:
        print('Yes')
    else:
        print('No')