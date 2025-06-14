# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.ans = 0
        def depth(node):# 一边dfs一边计算
            if not node:
                return 0
            l = depth(node.left)
            r = depth(node.right)
            self.ans = max(self.ans,l+r)
            return max(l,r)+1
        depth(root)
        return self.ans