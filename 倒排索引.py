from collections import defaultdict
n = int(input())
dic = defaultdict(set)
for i in range(1,n+1):
    for word in input().split()[1:]:
        dic[word].add(i)
for _ in range(int(input())):
    word = input()
    if word not in dic:
        print('NOT FOUND')
    else:
        print(*sorted(list(dic[word])))