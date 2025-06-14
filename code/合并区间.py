from typing import List
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        ans = []
        for start,final in intervals:
            if not ans:
                ans.append([start,final])
            else:
                prestart,prefinal = ans[-1]
                if prestart<=start<=prefinal:
                    ans[-1][1] = max(final,prefinal)
                else:
                    ans.append([start,final])
        return ans