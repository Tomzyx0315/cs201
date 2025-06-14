from collections import deque
while True:
    n = int(input())
    if n==1:
        print(1)
        continue
    if n==0:
        break
    queue = deque([10,11])
    modlist = set()
    modlist.add(1)
    while queue:
        num = queue.popleft()
        t = num%n
        if t==0:
            print(num)
            break
        else:
            if t not in modlist:
                modlist.add(t)
                queue.append(10*num+1)
                queue.append(10*num)