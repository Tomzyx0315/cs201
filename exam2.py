n = int(input())
nums = list(map(int,input().split()))
lines = [nums[0]]
def dfs(index):
    if index*2+2<n:
        lines.append(nums[index*2+2])
        dfs(index*2+2)
        lines.pop()
    if index*2+1<n:
        lines.append(nums[index*2+1])
        dfs(index*2+1)
        lines.pop()
    else:
        print(*lines)
dfs(0)
flag = True
couldbe = [True,True]# minheap,maxheap
def DFS(index):
    curr = nums[index]
    if index*2+1<n:
        left = nums[index*2+1]
        if curr<left:
            couldbe[1] = False
        if curr>left:
            couldbe[0] = False
        DFS(index*2+1)
    if index*2+2<n:
        right = nums[index*2+2]
        if curr<right:
            couldbe[1] = False
        if curr>right:
            couldbe[0] = False
        DFS(index*2+2)
DFS(0)
if couldbe[0]:
    print('Min Heap')
elif couldbe[1]:
    print('Max Heap')
else:
    print('Not Heap')