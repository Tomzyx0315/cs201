# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        def dfs(root):
            if not root:
                return
            left = root.left
            right = root.right
            root.left = None
            if left:
                root.right = left
                lastnode1 = dfs(left)
                if right:

            elif right:
                root.right = right
        dfs(root)