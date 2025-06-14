while True:
    m,n = map(int,input().split())
    if n+m==0:
        break
    left = m
    right = m
    ans = 0
    while True:
        if n>right:
            ans += right-left+1
            left = left*2
            right = right*2+1
        elif left<=n<=right:
            ans += n-left+1
            break
        else:
            break
    print(ans)