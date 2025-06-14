from collections import deque
n = int(input())
dic = {}
for _ in range(n):
    word = input()
    if word in dic:
        continue
    dic[word] = []
    for i in range(4):
        for j in range(26):
            newword1 = word[:i]+chr(ord('a')+j)+word[i+1:]
            newword2 = word[:i]+chr(ord('A')+j)+word[i+1:]
            if newword1 in dic:
                dic[newword1].append(word)
                dic[word].append(newword1)
            if newword2 in dic:
                dic[newword2].append(word)
                dic[word].append(newword2)
    
start,final = input().split()
queue = deque([start])
backtrack = {start:None}
while queue:
    word = queue.popleft()
    for neighbor in dic[word]:
        if neighbor==final:
            backtrack[neighbor] = word
            current = final
            ans = [current]
            while True:
                prev = backtrack[current]
                if not prev:
                    break
                ans.append(prev)
                current = prev
            print(' '.join(ans[::-1]))
            exit()
        if neighbor not in backtrack:
            backtrack[neighbor] = word
            queue.append(neighbor)
print('NO')