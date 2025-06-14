# 自己写的代码实在冗长，碰到这种复杂度高的就难以做对，还是看看灵神的代码吧！
class Solution:
    def minMoves(self, matrix: List[str]) -> int:
        from collections import defaultdict,deque
        inf = float('inf')
        # 这个特判一定要加！！
        if matrix[-1][-1] == '#':
            return -1

        m, n = len(matrix), len(matrix[0])
        pos = defaultdict(list)
        for i, row in enumerate(matrix):
            for j, c in enumerate(row):
                if c.isupper():
                    pos[c].append((i, j))

        DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dis = [[inf] * n for _ in range(m)]
        dis[0][0] = 0

        q = deque([(0, 0)])

        while q:
            x, y = q.popleft()
            d = dis[x][y]

            if x == m - 1 and y == n - 1:  # 到达终点
                return d

        # 为什么一碰到传送门就直接使用（或者即使不使用），并且删除对应的传送门？
        # 因为如果之后还要再用传送门的话，就肯定比这次慢了
            c = matrix[x][y]
            if c in pos:
                # 使用所有传送门
                for px, py in pos[c]:
                    if d < dis[px][py]:
                        dis[px][py] = d
                        q.appendleft((px, py))# 利用deque的appendleft处理传送门
                del pos[c]  # 避免重复使用传送门

            # 下面代码和普通 BFS 是一样的
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] != '#' and d + 1 < dis[nx][ny]:
                    dis[nx][ny] = d + 1
                    q.append((nx, ny))

        return -1