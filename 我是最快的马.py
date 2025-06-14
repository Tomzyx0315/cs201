from collections import deque
sx,sy = map(int,input().split())
fx,fy = map(int,input().split())
m = int(input())
others = set()
for _ in range(m):
    x,y = map(int,input().split())
    others.add((x,y))
steps = [(2,-1,1,0),(2,1,1,0),(1,2,0,1),(-1,2,0,1),(-2,1,-1,0),(-2,-1,-1,0),(-1,-2,0,-1),(1,-2,0,-1)]
queue = deque([(sx,sy,0)])
backtrack = [[None]*11 for _ in range(11)]
times = [[float('inf')]*11 for _ in range(11)]
times[sx][sy]=0
flag = False
'''
注意这里不能用in_queue判断！因为要找到所有最短路径数目！
in_queue = set()
in_queue.add((sx,sy))
'''
number_of_routes = 0
while queue and (not flag):
    for i in range(len(queue)):
        x,y,time = queue.popleft()# 是popleft补药写错啦！
        if x==fx and y==fy:
            number_of_routes+=1
            flag = True
        for dx,dy,bx,by in steps:
            nx,ny,tx,ty = x+dx,y+dy,x+bx,y+by
            if 0<=nx<11 and 0<=ny<11 and ((nx,ny) not in others) and ((time+1)<=times[nx][ny]):
                if not (0<=tx<11 and 0<=ty<11 and (tx,ty) in others):
                    queue.append((nx,ny,time+1))
                    backtrack[nx][ny]=(x,y)
                    times[nx][ny]=time+1
                    # 各种更新都不要漏掉！！！
if number_of_routes!=1:
    print(number_of_routes)
else:
    ans = []
    jx,jy = fx,fy
    while True:
        ans.append('('+str(jx)+','+str(jy)+')')
        if not backtrack[jx][jy]:
            break
        else:
            jx,jy = backtrack[jx][jy]
    print('-'.join(ans[::-1]))