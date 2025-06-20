# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def midorder(node):
            if not node:
                return []
            return midorder(node.left)+[node.val]+midorder(node.right)
        s = midorder(root)
        return s[k-1]