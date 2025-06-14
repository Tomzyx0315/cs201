import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
nums = [int(i) for i in data[index:index+n]]
ans = []
used = set()
visited = {}# 考虑重复关键字！！！！！
def check(index):
    i = 0
    while True:
        i+=1
        newindex1=(index+i**2)%m
        if newindex1 not in used:
            return newindex1
        newindex2=(index-i**2)%m
        if newindex2 not in used:
            return newindex2
for num in nums:
    if num in visited:
        ans.append(visited[num])
        continue
    index = num%m
    if index not in used:
        used.add(index)
        ans.append(index)
        visited[num]=index
    else:
        newindex = check(index)
        used.add(newindex)
        ans.append(newindex)
        visited[num]=newindex
print(' '.join(list(map(str,ans))))