class Solution:
    def hIndex(self, citations: List[int]) -> int:
        citations = sorted(citations,reverse=True)
        n = len(citations)
        for i in range(n,0,-1):
            if citations[i-1]>=i:
                return i
        return 0