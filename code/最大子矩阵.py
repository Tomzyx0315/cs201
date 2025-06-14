n = int(input())
numbers = []
MAX = float('-inf')
while True:
    try:
        numbers+=list(map(int,input().split()))
    except EOFError:
        break
numbers = [numbers[i*n:(i+1)*n] for i in range(n)]
for left in range(n):
    for right in range(left,n):
        processed = [sum(numbers[i][left:right+1]) for i in range(n)]
        dp = [0]*n
        dp[0]=processed[0]
        for i in range(1,n):
                dp[i]=max(dp[i-1]+processed[i],processed[i])
        MAX = max(MAX,max(dp))
print(MAX)