n = int(input())
nums = []
tot = set()
for _ in range(n):
    newset = set(map(int,input().split()[1:]))
    nums.append(newset)
    tot = tot.union(newset)
for _ in range(int(input())):
    check = list(map(int,input().split()))
    ans = tot
    for i in range(n):
        if check[i]==1:
            ans = ans.intersection(nums[i])
        elif check[i]==-1:
            ans = ans.difference(nums[i])
    if not ans:
        print('NOT FOUND')
    else:
        print(*sorted(list(ans)))