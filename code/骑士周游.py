n = int(input())
sr,sc = map(int,input().split())
steps = [(1,2),(2,1),(1,-2),(2,-1),(-1,2),(-2,1),(-1,-2),(-2,-1)]
def is_valid(x,y):
    return 0<=x<n and 0<=y<n and backtrack[x][y]
def get_degree(x,y):
    return sum([1 for dx,dy in steps if is_valid(x+dx,y+dy)])
backtrack = [[1]*n for _ in range(n)]
flag = False
backtrack[sr][sc]=0
def dfs(x,y,l):
    if l==n*n-1:
        print('success')
        exit()
    route = []
    for dx,dy in steps:
        nx,ny = x+dx,y+dy
        if is_valid(nx,ny):
            route.append((get_degree(nx,ny),nx,ny))
    route.sort()
    for degree,nx,ny in route:
        backtrack[nx][ny]=0
        dfs(nx,ny,l+1)
        backtrack[nx][ny]=1
dfs(sr,sc,0)
print('fail')