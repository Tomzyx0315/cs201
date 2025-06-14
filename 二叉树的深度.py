n = int(input())
is_root = [True]*(n+1)
tree = {i:[] for i in range(1,n+1)}
for i in range(1,n+1):
    a,b = map(int,input().split())
    tree[i]=[a,b]
    if a!=-1:
        is_root[a]=False
    if b!=-1:
        is_root[b]=False#非空判断坑！
def count(node):
    if node == -1:
        return 0
    return 1+max(count(tree[node][0]),count(tree[node][1]))
for i in range(1,n+1):
    if is_root[i]:
        print(count(i))