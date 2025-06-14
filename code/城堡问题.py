m = int(input())
n = int(input())
castle = [list(map(int,input().split())) for _ in range(m)]
def check(num):
    cant = []
    cant = list(bin(num)[2:])
    cant = ['0']*(4-len(cant))+cant
    return cant
steps = [(1,0),(0,1),(-1,0),(0,-1)]
def dfs(x,y):
    global tot
    cant = check(castle[x][y])
    for k in range(4):
        if cant[k]=='0':
            dx,dy = steps[k]
            nx,ny = x+dx,y+dy
            if not visited[nx][ny]:
                visited[nx][ny]=1
                tot+=1
                dfs(nx,ny)
MAX = 1
visited = [[0]*n for _ in range(m)]
roomnum = 0
for i in range(m):
    for j in range(n):
        if not visited[i][j]:
            roomnum+=1
            tot = 1
            visited[i][j]=1
            dfs(i,j)
            MAX = max(MAX,tot)
print(roomnum)
print(MAX)