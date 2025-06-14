import math
a = list(map(float,input().split()))
a.sort(reverse=True)
n = len(a)
b = a[-1+math.ceil(n*0.6)]
def f(c):
    c = c/1000000000
    return c*b+1.1**(c*b)
lo = 0
hi = 1000000000
while lo < hi-1:
    mid = (lo+hi)//2
    if f(mid)>=85:
        hi = mid
    else:
        lo = mid
if f(lo)>=85:
    print(lo)
else:
    print(hi)