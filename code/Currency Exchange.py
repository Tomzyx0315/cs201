# Bellman-ford算法的活用！
n,m,s,v = input().split()
n,m,s = int(n),int(m),int(s)
v = float(v)
edges = []
for _ in range(m):
    a,b,r1,c1,r2,c2 = input().split()
    a,b = int(a),int(b)
    r1,c1,r2,c2 = float(r1),float(c1),float(r2),float(c2)
    edges.append((a,b,r1,c1))
    edges.append((b,a,r2,c2))

max_amount = [0.0] * (n + 1)
max_amount[s] = v

for i in range(n-1):
    for a,b,r,c in edges:
        maxb = (max_amount[a]-c)*r
        max_amount[b]=max(maxb,max_amount[b])
flag = False
for a,b,r,c in edges:
    maxb = (max_amount[a]-c)*r
    if maxb>max_amount[b]:
        flag = True
if flag:
    print('YES')
else:
    print('NO')