n = int(input())
from collections import defaultdict
class Node:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
def preorder_traversal(root):
    if not root:
        return ''
    return root.val + preorder_traversal(root.left) + preorder_traversal(root.right)
def inorder_traversal(root):
    if not root:
        return ''
    return inorder_traversal(root.left) + root.val + inorder_traversal(root.right)
def postorder_traversal(root):
    if not root:
        return ''
    return postorder_traversal(root.left) + postorder_traversal(root.right) + root.val
for _ in range(n):
    state = 'left'
    to_be_len = 2
    tree = defaultdict(list)
    while True:
        s = input()
        if s=='0':
            break
        layer = len(s)-1
        if layer == 0:
            root = Node(s)
            tree[0] = [root]
            continue
        if state == 'right':
            node = Node(s[-1])
            prenode = tree[layer-1].pop()
            prenode.right = node
            tree[layer].append(node)
            state = 'left'
            to_be_len = layer+2
        else:
            if to_be_len != len(s):
                tree[to_be_len-2].pop()
                node = Node(s[-1])
                prenode = tree[layer-1][-1]
                prenode.right = node
                tree[layer].append(node)
                state = 'left'
                to_be_len = layer+2
                continue
            if s[-1]=='*':
                state = 'right'
            else:
                node = Node(s[-1])
                prenode = tree[layer-1][-1]
                prenode.left = node
                tree[layer].append(node)
                to_be_len = layer+2
            

    print(preorder_traversal(root))
    print(postorder_traversal(root))
    print(inorder_traversal(root))
    if _ != n-1:
        print()