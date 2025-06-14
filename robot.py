from collections import deque

def is_valid(x,y):
        if 0<=x<m+1 and 0<=y<n+1 and track[x][y]:
            return True
        return False

steps = [(-1,0),(0,1),(1,0),(0,-1)]
faces = {'north':0,'east':1,'south':2,'west':3}
while True:
    m,n = map(int,input().split())
    if m+n==0:
        break
    grid = [list(map(int,input().split())) for _ in range(m)]
    b1,b2,e1,e2,direction = input().split()
    b1,b2,e1,e2 = int(b1),int(b2),int(e1),int(e2)
    direction = faces[direction]
    track = [[1]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i==0 or j==0 or i==m or j ==n :
                track[i][j]=0
            elif grid[i][j] or grid[i-1][j-1] or grid[i][j-1] or grid[i-1][j]:
                track[i][j]=0
    if (not track[b1][b2]) or (not track[e1][e2]):
        print(-1)
        continue
    queue = deque([(b1,b2,direction,0)])
    in_queue = set()
    in_queue.add((b1,b2,direction))
    while queue:
        x,y,d,t =  queue.popleft()
        if x==e1 and y==e2:
            print(t)
            break
        if (x,y,(d+1)%4) not in in_queue:
            queue.append((x,y,(d+1)%4,t+1))
            in_queue.add((x,y,(d+1)%4))
        if (x,y,(d-1)%4) not in in_queue:
            queue.append((x,y,(d-1)%4,t+1))
            in_queue.add((x,y,(d-1)%4))
        dx,dy = steps[d]
        for i in range(3):
            x+=dx
            y+=dy
            if not is_valid(x,y):
                break
            else:
                if (x,y,d) not in in_queue:
                    queue.append((x,y,d,t+1))
                    in_queue.add((x,y,d))
    else:
        print(-1)