n = int(input())
nums = list(map(int,input().split()))
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
def maketree(nums):
    node  = TreeNode(nums[0])
    if len(nums)==1:
        return node
    for i in range(1,len(nums)):
        if nums[i]>nums[0]:
            break
    else:
        i=len(nums)
    left = nums[1:i]
    right = nums[i:]
    if left:
        node.left = maketree(left)
    if right:
        node.right = maketree(right)
    return node
root = maketree(nums)
def postorder(node):
    if not node:
        return []
    return postorder(node.left)+postorder(node.right)+[node.val]
ans = postorder(root)
print(' '.join(list(map(str,ans))))