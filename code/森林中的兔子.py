class Solution:
    def numRabbits(self, answers: List[int]) -> int:
        set1 = {}
        cnt = 0
        for num in answers:
            if num in set1:
                set1[num]+=1
                if set1[num]>num+1:
                    set1[num]=1
                    cnt+=(num+1)
            else:
                set1[num]=1
                cnt+=(num+1)
        return cnt