from typing import List
class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        m = len(grid[0])
        MOD = 12345
        ans = [[0]*m for _ in range(n)]

        suf = 1# 表示从下一个元素到结尾的乘积
        for i in range(n-1,-1,-1):
            for j in range(m-1,-1,-1):
                ans[i][j] = suf
                suf = suf*grid[i][j]%MOD

        pre = 1
        for i in range(n):
            for j in range(m):
                ans[i][j] = ans[i][j]*pre%MOD
                pre = pre*grid[i][j]%MOD
        return ans