from heapq import *
for _ in range(int(input())):
    m,n = map(int,input().split())
    seq1 = sorted(map(int,input().split()))
    for __ in range(m-1):
        seq2 = sorted(map(int,input().split()))
        min_heap = [(seq1[0]+seq2[i],0,i) for i in range(n)]
        heapify(min_heap)
        result = []
        for ___ in range(n):
            cur,i,j = heappop(min_heap)
            result.append(cur)
            if i+1 < n:
                heappush(min_heap,(seq1[i+1]+seq2[j],i+1,j))
        seq1 = result[::]
    print(*seq1)