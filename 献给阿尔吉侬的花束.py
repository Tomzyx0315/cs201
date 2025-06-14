from collections import deque
steps = [(-1,0),(1,0),(0,1),(0,-1)]
for _ in range(int(input())):
    r,c = map(int,input().split())
    maze = [input() for __ in range(r)]
    flag1 = False
    for p in range(r):
        for q in range(c):
            if maze[p][q]=='S':
                i,j = p,q
                
    queue = deque([(i,j,0)])
    in_queue = set((i,j))
    flag2 = True
    while queue:
        if not flag2:
            break
        x,y,t = queue.popleft()
        for dx, dy in steps:
            nx,ny = x+dx,y+dy
            if 0<=nx<r and 0<=ny<c and maze[nx][ny]!='#' and (nx,ny) not in in_queue:
                queue.append((nx,ny,t+1))
                in_queue.add((nx,ny))
                if maze[nx][ny]=='E':
                    print(t+1)
                    flag2 = False
                    break
    else:
        print('oop!')