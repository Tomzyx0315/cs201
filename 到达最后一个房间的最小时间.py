class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        distance = [[float('inf')]*m for _ in range(n)]
        steps = [(0,1),(0,-1),(1,0),(-1,0)]
        import heapq
        q = [(0,0,0)]
        while q:
            t,x,y = heapq.heappop(q)
            if x==n-1 and y==m-1:
                return t
            for dx,dy in steps:
                nx,ny = x+dx,y+dy
                if 0<=nx<n and 0<=ny<m:
                    nt = max(t,moveTime[nx][ny])+1
                    if nt<distance[nx][ny]:
                        heapq.heappush(q,(nt,nx,ny))
                        distance[nx][ny]=nt