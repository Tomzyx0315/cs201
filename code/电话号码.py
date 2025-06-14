class TrieNode:
    def __init__(self,val=None):
        self.val = val
        self.children = {}
        self.EOF = False
class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self,nums):
        node = self.root
        for char in nums:
            if char not in node.children:
                newnode = TrieNode(char)
                node.children[char]=newnode
                node = newnode
            else:
                node = node.children[char]
            if node.EOF:
                return False
        node.EOF = True
        return len(node.children)==0
    
for _ in range(int(input())):
    flag = True
    trie = Trie()
    for __ in range(int(input())):
        num = input()
        if not flag:
            continue
        flag = trie.insert(num)
    if flag:
        print("YES")
    else:
        print("NO")