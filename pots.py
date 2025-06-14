from collections import deque
a,b,c = map(int,input().split())
queue = deque([(0,0,0)])
backtrack = [[None]*(b+1) for _ in range(a+1)]
backtrack[0][0]='start'
while queue:
    i,j,d = queue.popleft()
    if i==c or j==c:
        ans = []
        while backtrack[i][j]!='start':
            ans.append(backtrack[i][j][0])
            i,j = backtrack[i][j][1],backtrack[i][j][2]
        print(len(ans))
        for line in ans[::-1]:
            print(line)
        break
    if not backtrack[i][0]:
        queue.append((i,0,d+1))
        backtrack[i][0]=('DROP(2)',i,j)
    if not backtrack[0][j]:
        queue.append((0,j,d+1))
        backtrack[0][j]=('DROP(1)',i,j)
    if not backtrack[i][b]:
        queue.append((i,b,d+1))
        backtrack[i][b]=('FILL(2)',i,j)
    if not backtrack[a][j]:
        queue.append((a,j,d+1))
        backtrack[a][j]=('FILL(1)',i,j)
    # pour 1 to 2
    if (b-j)>=i:
        x,y = 0,i+j
    else:
        x,y = i-(b-j),b
    if not backtrack[x][y]:
        queue.append((x,y,d+1))
        backtrack[x][y]=('POUR(1,2)',i,j)
    if (a-i)>=j:
        x,y = i+j,0
    else:
        x,y = a,j-(a-i)
    if not backtrack[x][y]:
        queue.append((x,y,d+1))
        backtrack[x][y]=('POUR(2,1)',i,j)
else:
    print('impossible')