class Treenode:
    def __init__(self,val,left=None,right=None):
        self.val=val
        self.children = []

def order1(root):
    if not root:
        return
    print(root.val,end='')
    for child in root.children:
        order1(child)

def order2(root):
    if not root:
        return
    for child in root.children:
        order2(child)
    print(root.val,end='')

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
        stack[-1].children = assisted[::-1]
        i+=1
    else:
        stack.append(Treenode(s[i]))
        i+=1
index = 0
root = stack[0]
order1(root)
print()
order2(root)