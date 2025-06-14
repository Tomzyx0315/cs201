class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        from collections import deque
        n = len(grid)
        steps = [(0,1),(0,-1),(1,0),(-1,0)]
        belonging = [[-1]*n for _ in range(n)]
        cnt = -1
        area = []
        for i in range(n):
            for j in range(n):
                if grid[i][j]==1 and belonging[i][j]==-1:
                    cnt+=1
                    tot=0
                    queue = deque([(i,j)])
                    belonging[i][j]=cnt
                    while queue:
                        x,y = queue.popleft()
                        tot+=1
                        for dx,dy in steps:
                            nx,ny = x+dx,y+dy
                            if 0<=nx<n and 0<=ny<n and grid[nx][ny] and belonging[nx][ny]==-1:
                                queue.append((nx,ny))
                                belonging[nx][ny]=cnt
                    area.append(tot)
        if cnt == -1:
            return 1
        MAX = max(area)
        for i in range(n):
            for j in range(n):
                if grid[i][j]==0:
                    prepare = set()
                    for dx,dy in steps:
                        nx,ny = i+dx,j+dy
                        if 0<=nx<n and 0<=ny<n and belonging[nx][ny]!=-1:
                            prepare.add(belonging[nx][ny])
                    MAX = max(MAX,1+sum([area[k] for k in prepare]))
        return MAX