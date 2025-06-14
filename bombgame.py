a,b,k = map(int,input().split())
matrix = [[0]*b for _ in range(a)]
for _ in range(k):
    r,s,p,t = map(int,input().split())
    r,s = r-1,s-1
    for i in range(r-(p-1)//2,r+(p+1)//2):
        for j in range(s-(p-1)//2,s+(p+1)//2):
            if 0<=i<a and 0<=j<b:
                if t:
                    matrix[i][j]+=1
                else:
                    matrix[i][j]-=1
cnt = 0
MAX = max([max(matrix[i]) for i in range(a)])
for i in range(a):
    for j in range(b):
        if matrix[i][j]==MAX:
            cnt+=1
print(cnt)