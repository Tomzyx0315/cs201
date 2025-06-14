n,k = map(int,input().split())
a = [float(input())*100 for _ in range(n)]
def check(l):
    tot = sum([b//l for b in a])
    if tot>=k:
        return True
    return False
lo = 0
hi = max(a)
if not check(1):
    print('0.00')
else:
    while lo<hi-1:
        mid = (lo+hi)//2
        if check(mid):
            lo = mid
        else:
            hi = mid-1
    if lo == hi-1:
        if check(hi):
            ans = hi
        else:
            ans = lo
    else:
        ans = lo
    print(f"{ans/100:.2f}")