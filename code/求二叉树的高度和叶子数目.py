n = int(input())
is_root = [True]*n
tree = {i:[] for i in range(n)}
cnt = 0
for i in range(n):
    a,b = map(int,input().split())
    tree[i]=[a,b]
    if a!=-1:
        is_root[a]=False
    if b!=-1:
        is_root[b]=False#非空判断坑！
    if a==b==-1:
        cnt+=1
def count(node):
    if node == -1:
        return -1
    return 1+max(count(tree[node][0]),count(tree[node][1]))
for i in range(n):
    if is_root[i]:
        print(count(i),cnt)