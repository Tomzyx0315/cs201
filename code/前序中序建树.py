# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        def dfs(preorder,inorder):
            root = TreeNode(preorder[0])
            index = inorder.index(preorder[0])
            leftin = inorder[:index]
            rightin = inorder[index+1:]
            leftpre = preorder[1:index+1]
            rightpre = preorder[index+1:]
            if leftin and leftpre:
                root.left = dfs(leftpre,leftin)
            if rightin and rightpre:
                root.right = dfs(rightpre,rightin)
            return root
        return dfs(preorder,inorder)