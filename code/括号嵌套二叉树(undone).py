class Treenode:
    def __init__(self,val,left=None,right=None):
        self.val=val
        self.left = None
        self.right = None

def order1(root):
    if not root:
        return
    if root.val == '*':
        return
    print(root.val,end='')
    order1(root.left)
    order1(root.right)

def order2(root):
    if not root:
        return
    if root.val == '*':
        return
    order2(root.left)
    print(root.val,end='')
    order2(root.right)
for _ in range(int(input())):
    s = input()
    stack = []
    i = 0
    while i<len(s):
        if s[i]==',':
            i+=1
            continue
        if s[i]=='(':
            stack.append(s[i])
            i+=1
        elif s[i]==')':
            assisted = []
            while stack[-1]!='(':
                assisted.append(stack.pop())
            stack.pop()
            stack[-1].left = assisted[1]
            stack[-1].right = assisted[0]
            i+=1
        else:
            stack.append(Treenode(s[i]))
            i+=1
    index = 0
    root = stack[0]
    order1(root)
    print()
    order2(root)
    print()