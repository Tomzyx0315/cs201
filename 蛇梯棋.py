class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        n = len(board)
        mapping = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                mapping[i][j] = (j+1)+n*i
        for i in range(1,n,2):
            mapping[i] = mapping[i][::-1]
        mapping = mapping[::-1]
        ladders = {}
        for i in range(n):
            for j in range(n):
                if board[i][j]!=-1:
                    ladders[mapping[i][j]]=board[i][j]
        from collections import deque
        queue = deque([(1,0)])
        in_queue = set()
        in_queue.add(1)
        while queue:
            curr,distance = queue.popleft()
            if curr == n*n:
                return distance
            for i in range(curr+1,1+min(curr+6,n*n)):
                if i in ladders:
                    dest = ladders[i]
                else:
                    dest = i
                if dest not in in_queue:
                    in_queue.add(dest)
                    queue.append((dest,distance+1))
        return -1