# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def maketree(a):
            if not a:
                return None
            n = len(a)
            index = n//2
            node = TreeNode(a[index])
            node.left = maketree(a[:index])
            node.right = maketree(a[index+1:])
            return node
        return maketree(nums)