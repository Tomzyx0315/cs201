class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
        self.pre = None
        self.loc = -1
for __ in range(int(input())):
    n,m = map(int,input().split())
    tree = [TreeNode(i) for i in range(n)]
    for _ in range(n):
        x,y,z = map(int,input().split())
        if y != -1:
            tree[x].left = tree[y]
            tree[y].pre = tree[x]
            tree[y].loc = 0
        if z != -1:
            tree[x].right = tree[z]
            tree[z].pre = tree[x]
            tree[z].loc = 1
    for _ in range(m):
        s = input()
        if s[0]=='1':
            s = s.split()
            x,y = int(s[1]),int(s[2])
            x,y = tree[x],tree[y]
            rootx = x.pre
            rooty = y.pre
            if x.loc:
                rootx.right = y
            else:
                rootx.left = y
            if y.loc:
                rooty.right = x
            else:
                rooty.left = x
            x.pre, y.pre = rooty, rootx
            x.loc, y.loc = y.loc, x.loc# x,y的重置反而容易忘掉

        if s[0]=='2':
            s = s.split()
            root = tree[int(s[1])]
            while root.left:
                root = root.left
            print(root.val)

'''
列表模拟反而比较简单
def swap(x, y):
    tree[loc[x][0]][loc[x][1]] = y
    tree[loc[y][0]][loc[y][1]] = x
    loc[x], loc[y] = loc[y], loc[x]


for _ in range(int(input())):
    n, m = map(int, input().split())
    tree = {}
    loc = [[] for _ in range(n)]
    for _ in range(n):
        a, b, c = map(int, input().split())
        tree[a] = [b, c]
        loc[b], loc[c] = [a, 0], [a, 1]
    for _ in range(m):
        op = list(map(int, input().split()))
        if op[0] == 1:
            swap(op[1], op[2])
        else:
            cur = op[1]
            while tree[cur][0] != -1:
                cur = tree[cur][0]
            print(cur)
'''