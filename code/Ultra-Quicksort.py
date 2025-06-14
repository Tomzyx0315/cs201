from bisect import bisect_left
while True:
    n = int(input())
    if n==0:
        break
    v = [int(input()) for _ in range(n)]
    def merge(v):
        if (not v) or len(v)==1:
            return 0
        index = (len(v))//2-1
        v1 = v[:index+1]
        v2 = v[index+1:]
        a = merge(v1)
        b = merge(v2)
        v1.sort()
        v2.sort()
        cnt = 0
        for vx in v1:
            loc = bisect_left(v2,vx)
            cnt+=loc
        return a+b+cnt
    print(merge(v))