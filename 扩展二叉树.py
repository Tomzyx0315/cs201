class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
def midorder(node):
    if not node:
        return ''
    return midorder(node.left)+node.val+midorder(node.right)
def postorder(node):
    if not node:
        return ''
    return postorder(node.left)+postorder(node.right)+node.val

s = input()
index = -1
def maketree():
    global index
    index+=1
    if s[index]=='.':
        return None
    node = TreeNode(s[index])
    left = maketree()
    right = maketree()
    if left:
        node.left = left
    if right:
        node.right = right
    return node
root = maketree()
print(midorder(root))
print(postorder(root))