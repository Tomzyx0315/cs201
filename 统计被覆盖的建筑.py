from typing import List
class Solution:
    def countCoveredBuildings(self, n: int, buildings: List[List[int]]) -> int:
        rows = [0]*n
        columns = [0]*n
        rowstart = [n]*n
        columnstart = [n]*n
        rowend = [-1]*n
        columnend = [-1]*n
        for x,y in buildings:
            x,y=x-1,y-1
            rows[x]+=1
            columns[y]+=1
            rowstart[x]=min(rowstart[x],y)
            columnstart[y]=min(columnstart[y],x)
            rowend[x]=max(rowend[x],y)
            columnend[y]=max(columnend[y],x)
        cnt = 0
        for x,y in buildings:
            x,y=x-1,y-1
            if rows[x]>=3 and columns[y]>=3 and rowstart[x]!=y and rowend[x]!=y and columnstart[y]!=x and columnend[y]!=x:
                cnt+=1
        return cnt