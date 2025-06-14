class Spreadsheet:

    def __init__(self, rows: int):
        self.grid = [[0]*26 for _ in range(rows)]

    def setCell(self, cell: str, value: int) -> None:
        j = ord(cell[0])-ord('A')
        i = int(cell[1:])-1
        self.grid[i][j]=value

    def resetCell(self, cell: str) -> None:
        j = ord(cell[0])-ord('A')
        i = int(cell[1:])-1
        self.grid[i][j]=0

    def getValue(self, formula: str) -> int:
        index = formula.find('+')
        str1 = formula[1:index]
        str2 = formula[index+1:]
        if 'A'<=str1[0]<='Z':
            j = ord(str1[0])-ord('A')
            i = int(str1[1:])-1
            value1 = self.grid[i][j]
        else:
            value1 = int(str1)
        if 'A'<=str2[0]<='Z':
            j = ord(str2[0])-ord('A')
            i = int(str2[1:])-1
            value2 = self.grid[i][j]
        else:
            value2 = int(str2)
        return value1+value2
        


# Your Spreadsheet object will be instantiated and called as such:
# obj = Spreadsheet(rows)
# obj.setCell(cell,value)
# obj.resetCell(cell)
# param_3 = obj.getValue(formula)