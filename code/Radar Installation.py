case = 0
while True:
    n,d = map(int,input().split())
    if n==d==0:
        break
    case+=1
    intervals = []
    flag = True
    for _ in range(n):
        x,y = map(int,input().split())
        if d<abs(y):
            flag = False
        else:
            delta = (d**2-y**2)**0.5
            intervals.append([x-delta,x+delta])
    if not flag:
        print(f'Case {case}: -1')
        input()
        continue
    cnt = 0
    ans = []
    intervals.sort()
    for start,final in intervals:
        if not ans:
            cnt+=1
            ans.append([start,final])
        else:
            prestart,prefinal = ans[-1]
            if prestart<=start<=prefinal:
                ans[-1][0] = start
                ans[-1][1] = min(prefinal,final)
            else:
                cnt+=1
                ans.append([start,final])

    print(f'Case {case}: {cnt}')
    input()