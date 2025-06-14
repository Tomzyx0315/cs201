# pylint: skip-file
def dfs(x,y,l):
    global flag
    if flag:
        return
    if l==p*q:
        flag = True
        ans = [(x,y)]
        while board[x][y]!='start':
            x,y = board[x][y]
            ans.append((x,y))
        ans = ans[::-1]
        for x,y in ans:
            z = chr(ord('A')+x)
            print(f'{z}{y+1}',end = '')
        print()
        return
    for dx,dy in steps:
        nx,ny = x+dx,y+dy
        if 0<=nx<q and 0<=ny<p and not board[nx][ny]:
            board[nx][ny]=(x,y)
            dfs(nx,ny,l+1)
            board[nx][ny]=None
t = int(input())
for _ in range(t):
    print(f'Scenario #{_+1}:')
    flag = False
    p,q = map(int,input().split())
    board = [[None]*p for __ in range(q)]
    steps = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
    for i in range(q):
        for j in range(p):
            if not flag:
                board[i][j]='start'
                dfs(i,j,1)
    if not flag:
        print('impossible')
    if _ != t-1:
        print()