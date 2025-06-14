from collections import deque
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m,n = len(grid),len(grid[0])
        times = [[float('inf')]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if grid[i][j]==2:
                    times[i][j]=0
                elif grid[i][j]==0:
                    times[i][j]=0
        visited = [[False]*n for _ in range(m)]
        steps = [(1,0),(-1,0),(0,1),(0,-1)]
        queue = deque([])
        for i in range(m):
            for j in range(n):
                if grid[i][j]==2 and not visited[i][j]:
                    visited[i][j]=True
                    queue.append((i,j,0))
        while queue:
            x,y,t = queue.popleft()
            for dx,dy in steps:
                nx,ny = x+dx,y+dy
                if 0<=nx<m and 0<=ny<n and not visited[nx][ny] and grid[nx][ny]==1:
                    visited[nx][ny]=True
                    times[nx][ny]=t+1
                    queue.append((nx,ny,t+1))
        MAX = max([max(line) for line in times])
        if MAX == float('inf'):
            return -1
        return MAX