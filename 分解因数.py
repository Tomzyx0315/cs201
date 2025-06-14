from functools import lru_cache
@lru_cache
def divide(a,l):
    if a==1:
        return 1
    if a<=l:
        return 0
    tot = 0
    for i in range(l+1,a+1):
        if a%i==0:
            tot+=divide(a//i,i-1)
    return tot
for _ in range(int(input())):
    a = int(input())
    print(divide(a,1))