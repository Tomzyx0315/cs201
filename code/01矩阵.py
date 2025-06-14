from collections import deque
from typing import List
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m,n = len(mat),len(mat[0])
        queue = deque([])
        ans = [[0]*n for _ in range(m)]
        steps = [(0,1),(0,-1),(1,0),(-1,0)]
        visited = [[0]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if mat[i][j]==0:
                    for dx,dy in steps:
                        nx,ny = i+dx,j+dy
                        if 0<=nx<m and 0<=ny<n and mat[nx][ny]==1:
                            queue.append((nx,ny,1))
                            visited[nx][ny]=1
                            ans[nx][ny]=1
        while queue:
            x,y,d = queue.popleft()
            for dx,dy in steps:
                nx,ny = x+dx,y+dy
                if 0<=nx<m and 0<=ny<n and mat[nx][ny]==1 and not visited[nx][ny]:
                    queue.append((nx,ny,d+1))
                    ans[nx][ny]=d+1
                    visited[nx][ny]=1
        return ans