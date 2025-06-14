r,c = map(int,input().split())
h = [list(map(int,input().split())) for _ in range(r)]
height = []
for i in range(r):
    for j in range(c):
        height.append((h[i][j],i,j))
height.sort()
dp = [[0]*c for _ in range(r)]
steps = [(0,1),(0,-1),(1,0),(-1,0)]
for i in range(r*c):
    hi,x,y = height[i]
    MAX = 0
    for dx,dy in steps:
        nx,ny = x+dx,y+dy
        if 0<=nx<r and 0<=ny<c:
            if h[nx][ny]<h[x][y]:
                MAX = max(MAX,dp[nx][ny])
    dp[x][y]=MAX+1
print(max([max(line) for line in dp]))