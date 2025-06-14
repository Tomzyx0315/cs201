nums = list(map(int,input().split()))
# 记得查重
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
root = TreeNode(nums[0])
for i in range(1,len(nums)):
    num = nums[i]
    node = root
    while True:
        if num == node.val:
            break
        if num > node.val:
            if node.right:
                node = node.right
            else:
                node.right = TreeNode(num)
        else:
            if node.left:
                node = node.left
            else:
                node.left = TreeNode(num)
from collections import deque
queue = deque([root])
ans = []
while queue:
    node = queue.popleft()
    if node.left:
        queue.append(node.left)
    if node.right:
        queue.append(node.right)
    ans.append(str(node.val))
print(' '.join(ans))