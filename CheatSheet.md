## Cheat Sheet

```python
# pylint: skip-file
```

### 链表

判断两个list node的值是否相等用的是node.val

#### 反转链表

```python
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    pre = None
    current = head
    while current:
        next = current.next
        current.next = pre
        pre = current
        current = next
    return pre
```

#### 双向链表（LRU Cache：key->node字典）

```python
class ListNode:
    def __init__(self,key,value):
        self.key = key
        self.value = value
        self.pre = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = ListNode(0, 0)  # dummy head
        self.tail = ListNode(0, 0)  # dummy tail
        self.head.next = self.tail
        self.tail.pre = self.head
    def append(self,node):
        node.pre = self.tail.pre
        node.next = self.tail
        self.tail.pre.next = node
        self.tail.pre = node

    def remove(self,node):
        node.pre.next = node.next
        node.next.pre = node.pre
        
    def pop(self):
        if self.head.next == self.tail:
            return None
        node = self.head.next
        self.remove(node)
        return node.key
```

### Tree

核心思路是递归

#### 最小生成树

```python
# kruscal算法
# 基于disjoint set，适用于加权无向图
# 它的基本思想是从最小权重的边开始，逐步添加到生成树中，同时避免形成环，直到构造出包含所有顶点的最小生成树
lines.sort()
disjoint = DisjointSet(n)
cnt = 0
tot = 0
for weight, start, final in lines:
    start = names[start]
    final = names[final]
    if cnt == n-1:
        print(tot)
        exit()
    if disjoint.find(start)!=disjoint.find(final):
        cnt+=1
        tot+=weight
        disjoint.union(start,final)
print(tot)
```

MST算法背后的贪心算法不只适用于上面提到的最短路径问题，其他的变种也是适用的，比如最小化一条路径上的最大步长

### Graph

#### 拓扑排序（应用-判断有向图有没有环）

```python
# Kahn算法
queue = deque([])
topology = []
for i in range(1,n+1):
    if in_degree[i]==0:
        queue.append(i)
while queue:
    point = queue.popleft()
    topology.append(point)
    if point in graph:
        for nextpoint in graph[point]:
            in_degree[nextpoint]-=1
            if in_degree[nextpoint]==0:
                queue.append(nextpoint)
if len(topology)==n:
    print('No')
else:
    print('Yes')
```

#### 无向图有没有环

```python
# 比较恶心的一点是这个图可能是不连通的，所以要判断整个图有没有环需要把所有的联通部分都走一遍，这里就假设是个连通图
# 无向图的环的定义是不沿原路返回而形成的环，在遇到第二次入队的点时，如果这一步上一步并不是原路返回，那说明形成了环
flag = False
in_queue = set()
parent = {}
def bfs(firstpoint):
    global flag, in_queue
    queue = deque([firstpoint])
    in_queue.add(firstpoint)
    parent[firstpoint] = None
    while queue:
        point = queue.popleft()
        for nextpoint in graph[point]:
            if nextpoint not in in_queue:
                in_queue.add(nextpoint)
                queue.append(nextpoint)
                parent[nextpoint] = point# 记录上一步
            elif parent[point] != nextpoint:
                flag = True
```

### Dijkstra

```python
# 这个是plus版，加上了返回最短路径的部分
def minroute(start,final):
    if start == final:
        print(start)
        return
    distances = {}
    backtrack = {}
    for loc in locs:
        distances[loc] = float('inf')
    distances[start] = 0
    priority_queue = [(0,start)]
    while priority_queue:
        distance, x = heapq.heappop(priority_queue)
        for nextmove,d in graph[x].items():
            newdistance = distance+d
            if newdistance<distances[nextmove]:
                distances[nextmove]=newdistance
                backtrack[nextmove]=x
                heapq.heappush(priority_queue,(newdistance,nextmove))
    route = final
    nextpoint = final
    while True:
        lastpoint = backtrack[nextpoint]
        route = lastpoint+f'->({graph[lastpoint][nextpoint]})->'+route
        nextpoint = lastpoint
        if lastpoint==start:
            break
    print(route)
```

#### Dijkstra优化

```python
graph = [[] for _ in range(n)]# 不要用字典！要考虑到可能有重边
distance = [inf]*n
distance[0]=0
priority_queue = [(0,0)]
while priority_queue:
    curd,cur = heapq.heappop(priority_queue)
    if curd>distance[cur]:# 剪枝
        continue
    if cur == n-1:
        print(curd)
        exit()# 提前退出
    for next,d in graph[cur]:
        newd = curd+d
        if newd < distance[next]:
            distance[next] = newd
            heapq.heappush(priority_queue,(newd,next))
print(distance[n-1])
```

#### Bellman-ford

```python
dist = [float('inf')]*n
dist[src]=0

for _ in range(k + 1):  # 最多k+1条边
    new_dist = dist[:]  # 保持一份拷贝，防止使用当轮更新的结果
    for u, v, w in flights:
        if dist[u] != float('inf') and dist[u] + w < new_dist[v]:
            new_dist[v] = dist[u] + w
    dist = new_dist

return -1 if dist[dst] == float('inf') else dist[dst]
# 最多中转k次，也即，边长最多k+1；对于无限制的Bellman-ford算法，松弛次数改为V-1，V为节点数
'''
检查是否存在负权环
for u, v, w in graph:
    if dist[u] != float('inf') and dist[u] + w < dist[v]:
        print("图中存在负权环")
        return None
'''
```

#### 关键路径

```python
# 一个工程中某一步的时间是定死的-关键路径
# 如何判断一条边是不是关键路径？
# 前向计算，计算最早可能开始时间
# 后向计算，计算最晚可能开始时间
# 两者相等就是关键路径
from functools import lru_cache

n,m = map(int,input().split())
graphforward = {}
graphbackward = {}
lines = []
for _ in range(m):
    a,b,c = map(int,input().split())
    if a not in graphforward:
        graphforward[a] = {}
        graphforward[a][b]=c
    else:
        graphforward[a][b]=c
    if b not in graphbackward:
        graphbackward[b] = {}
        graphbackward[b][a]=c
    else:
        graphbackward[b][a]=c
    lines.append((a,b))

@ lru_cache
def forward(b,c):
    if b not in graphbackward:
        return 0
    else:
        MAX = float('-inf')
        for a in graphbackward[b]:
            MAX = max(graphforward[a][b]+forward(a,b),MAX)
    return MAX
FW = {}
for a,b in lines:
    FW[(a,b)] = forward(a,b)
maxtime = float('-inf')
for a,b in lines:
    maxtime = max(maxtime,FW[(a,b)]+graphforward[a][b])

@ lru_cache
def backward(a,b):
    if b not in graphforward:
        return maxtime-graphforward[a][b]
    else:
        MIN = float('inf')
        for c in graphforward[b]:
            MIN = min(MIN,backward(b,c)-graphforward[a][b])
    return MIN
BW = {}
for a,b in lines:
    BW[(a,b)] = backward(a,b)
lines.sort()
print(maxtime)
for a,b in lines:
    if FW[(a,b)]==BW[(a,b)]:
        print(a,b)
```

### Disjoint Set

#### 基本类

```python
class DisjointSet:
    def __init__(self, n):
        # 初始化父节点和秩
        self.parent = list(range(n))  # 每个元素的父节点初始化为自己
        self.rank = [0] * n  # 每个元素的秩初始化为0

    def find(self, x):
        # 查找元素x所在集合的代表元素（带路径压缩）
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        # 合并两个集合（按秩合并）
        rootX = self.find(x)
        rootY = self.find(y)
        
        if rootX != rootY:
            # 按秩合并
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
```

扩展：两个犯罪团伙，N个案件，每条信息给出由不同犯罪团伙犯下的两个案件，询问两个案件是否是同一个犯罪团伙所做

```python
n, m = map(int, input().split())
uf = UnionFind(2 * n)  # 初始化并查集，每个案件对应两个节点
for _ in range(m):
    operation, a, b = input().split()
    a, b = int(a) - 1, int(b) - 1
    if operation == "D":
        uf.union(a, b + n)  # a与b的对立案件合并
        uf.union(a + n, b)  # a的对立案件与b合并
    else:  # "A"
        if uf.find(a) == uf.find(b) or uf.find(a + n) == uf.find(b + n):
            print("In the same gang.")
        elif uf.find(a) == uf.find(b + n) or uf.find(a + n) == uf.find(b):
            print("In different gangs.")
        else:
            print("Not sure yet.")
```

扩展：三类动物，N个动物，环形食物链：同类，吃，被吃

```python
dsu = DisjointSet(n)

def find_disjoint_set(x):
    if x > n:
        return False
    return True

false_count = 0
for d, x, y in statements:
    if not find_disjoint_set(x) or not find_disjoint_set(y):
        false_count += 1
        continue
    if d == 1:  # X and Y are of the same type
        if dsu.find(x) == dsu.find(y + n) or dsu.find(x) == dsu.find(y + 2 * n):
            false_count += 1
        else:
            dsu.union(x, y)
            dsu.union(x + n, y + n)
            dsu.union(x + 2 * n, y + 2 * n)
    else:  # X eats Y
        if dsu.find(x) == dsu.find(y) or dsu.find(x + 2*n) == dsu.find(y):
            false_count += 1
        else: #[1,n] 区间表示同类，[n+1,2*n]表示x吃的动物，[2*n+1,3*n]表示吃x的动物
            dsu.union(x + n, y)
            dsu.union(x, y + 2 * n)
            dsu.union(x + 2 * n, y + n)

return false_count
```

### Trie

#### 基本类

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # 子节点，键为字符，值为TrieNode
        self.is_end_of_word = False  # 是否是一个单词的结尾

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word  # 只有完全匹配才返回True

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True  # 只要前缀存在就返回True

# 查是否一个字符串是另一个字符串前缀
class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self,nums):
        node = self.root
        for char in nums:
            if char not in node.children:
                newnode = TrieNode(char)
                node.children[char]=newnode
                node = newnode
            else:
                node = node.children[char]
            if node.EOF:
                return False
        node.EOF = True
        return len(node.children)==0
```

## heap

双堆实现数据流中位数

```python
from heapq import *
# 注意python的堆是最小堆，想要用最大堆记得取负！很难debug！
class MedianFinder:

    def __init__(self):
        self.minheap = []
        self.maxheap = []

    def addNum(self, num: int) -> None:
        heappush(self.maxheap, -num)
        
        # 确保最大堆堆顶 <= 最小堆堆顶
        if self.minheap and -self.maxheap[0] > self.minheap[0]:
            heappush(self.minheap, -heappop(self.maxheap))

        # 保证两个堆大小平衡，最多相差1
        if len(self.maxheap) > len(self.minheap) + 1:
            heappush(self.minheap, -heappop(self.maxheap))
        elif len(self.minheap) > len(self.maxheap):
            heappush(self.maxheap, -heappop(self.minheap))

    def findMedian(self) -> float:
        if len(self.maxheap)>len(self.minheap):
            return -self.maxheap[0]
        else:
            return (-self.maxheap[0]+self.minheap[0])/2
```

多路合并：拆分成两路合并，爆内存之后采取heapq优化

```python
min_heap = [(seq1[0]+seq2[i],0,i) for i in range(n)]
heapify(min_heap)
result = []
for ___ in range(n):
    cur,i,j = heappop(min_heap)
    result.append(cur)
    if i+1 < n:
        heappush(min_heap,(seq1[i+1]+seq2[j],i+1,j))
seq1 = result[::]
```

## stack

波兰表达式：* + 11.0 12.0 + 24.0 35.0

```python
s = input().split()[::-1]
stack = []
for i in s:
    if i in ['+','-','*','/']:
        a = stack.pop()
        b = stack.pop()
        stack.append(str(eval(a+i+b)))
    else:
        stack.append(i)
print(f'{float(stack[-1]):.6f}')
```

今日化学论文：d[2b[3a]c]

```python
s = input()
n = len(s)
ans = ''
numstack = []
stack = []
i = 0
while i < n:
    if s[i].isnumeric():
        nums = s[i]
        while i+1 < n:
            if s[i+1].isnumeric():
                i+=1
                nums+=s[i]
            else:
                numstack.append(int(nums))
                break
    elif s[i]=='[':
        stack.append(s[i])
    elif s[i]==']':
        assisted = ''
        while stack and stack[-1]!='[':
            assisted = stack.pop()+assisted
        stack.pop()
        stack.append(assisted*numstack.pop())
    else:
        stack.append(s[i])
    i+=1
#注意最外层不一定是括号！
print(''.join(stack))
```

中序表达式转后序表达式：(3)\*((3+4)*(2+3.5)/(4+5)) 

```python
def infix_to_postfix(expression):
    precedence = {'+':1, '-':1, '*':2, '/':2}
    stack = []
    postfix = []
    number = ''

    for char in expression:
        if char.isnumeric() or char == '.':
            number += char
        else:
            if number:
                num = float(number)
                postfix.append(int(num) if num.is_integer() else num)
                number = ''
            if char in '+-*/':
                while stack and stack[-1] in '+-*/' and precedence[char] <= precedence[stack[-1]]:
                    postfix.append(stack.pop())
                stack.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())
                stack.pop()

    if number:
        num = float(number)
        postfix.append(int(num) if num.is_integer() else num)

    while stack:
        postfix.append(stack.pop())

    return ' '.join(str(x) for x in postfix)

n = int(input())
for _ in range(n):
    expression = input()
    print(infix_to_postfix(expression))
```

柱形图中的最大矩形

```python
stack = []
heights = [0]+heights+[0]
rec = 0
for i in range(len(heights)):
    while stack and heights[stack[-1]]>heights[i]:
        height = heights[stack.pop()]
        w = i-stack[-1]-1
        rec = max(rec,height*w)
    stack.append(i)
return rec
```

接雨水

```python
def trap(self, height: List[int]) -> int:
    tot = 0
    stack = []
    n = len(height)
    i = 0
    while i<n:
        while stack and height[stack[-1]]<height[i]:
            index = stack.pop()
            h = height[index]
            if stack:
                tot += (min(height[stack[-1]],height[i])-h)*(i-stack[-1]-1)
        stack.append(i)
        i+=1
    return tot
```

有多少种合法的出栈顺序

```python
n = int(input())
catalan = [0] * (n + 1)
catalan[0] = 1  # C0 = 1

for i in range(1, n + 1):
    for j in range(i):
        catalan[i] += catalan[j] * catalan[i - 1 - j]
print(catalan[n])
```

## Sort

有时候对数据先预处理sort一下能极大简化思考

注意题中的不等关系是否包含等号

排序后回到原顺序：

```python
wrapup = list(enumerate(nums1))
wrapup.sort(key=lambda x:x[1])
```

## Tree

有时候不用OOP写法，用列表模拟更快

在DFS中同时进行多项操作：

```python
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.ans = 0
        def depth(node):# 一边dfs一边计算
            if not node:
                return 0
            l = depth(node.left)
            r = depth(node.right)
            self.ans = max(self.ans,l+r)
            return max(l,r)+1
        depth(root)
        return self.ans
```

层序bfs：

```python
while queue:
    for _ in range(len(queue)):
        node = queue.popleft()
```

抽象BST：搜索二维矩阵

```python
m = len(matrix)
n = len(matrix[0])
r = 0
c = m-1
while r<n and c>=0:
    if matrix[r][c]==target:
        return True
    if matrix[r][c]>target:
        c-=1
    else:
        r+=1
return False
```

## Graph

BFS的popleft()不要写成pop()!!!

BFS要找到所有最短路径数目不能用in_queue记录

```python
queue = deque([(sx,sy,0)])
times = [[float('inf')]*11 for _ in range(11)]
times[sx][sy]=0
flag = False
number_of_routes = 0
while queue and (not flag):
    for i in range(len(queue)):
        x,y,time = queue.popleft()# 是popleft补药写错啦！
        if x==fx and y==fy:
            number_of_routes+=1
            flag = True
        for dx,dy in steps:
            nx,ny = x+dx,y+dy
            if 0<=nx<11 and 0<=ny<11 and ((nx,ny) not in others) and ((time+1)<=times[nx][ny]):
                queue.append((nx,ny,time+1))
                times[nx][ny]=time+1
```

传送门：

为什么一碰到传送门就直接使用（或者即使不使用），并且删除对应的传送门？因为如果之后还要再用传送门的话，就肯定比这次慢了

```python
while q:
    x, y = q.popleft()
    d = dis[x][y]

    if x == m - 1 and y == n - 1:  # 到达终点
        return d

    c = matrix[x][y]
    if c in pos:
        # 使用所有传送门
        for px, py in pos[c]:
            if d < dis[px][py]:
                dis[px][py] = d
                q.appendleft((px, py))# 利用deque的appendleft处理传送门
        del pos[c]  # 避免重复使用传送门(这个主要是为了避免死循环，重复使用传送门肯定是更慢的)

    # 下面代码和普通 BFS 是一样的
    for dx, dy in DIRS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] != '#' and d + 1 < dis[nx][ny]:
            dis[nx][ny] = d + 1
            q.append((nx, ny))

return -1
```

拓扑排序：n==len(topology)说明无环（可以不连通），在此基础上cnt=0说明存在严格的不等关系

DFS的Wardsdorff优化：

```python
def get_degree(x,y):
    return sum([1 for dx,dy in steps if is_valid(x+dx,y+dy)])
route = []
for dx,dy in steps:
    nx,ny = x+dx,y+dy
    if is_valid(nx,ny):
        route.append((get_degree(nx,ny),nx,ny))
route.sort()
for degree,nx,ny in route:
    backtrack[nx][ny]=0
    dfs(nx,ny,l+1)
    backtrack[nx][ny]=1
```

拓扑排序+dp：重点在于确认信息流向！一个方向的依赖性是可以保证的，另一个是不可以保证的！

### 前后缀和

每次只对某一个数字进行操作，求对每个数字操作的结果

```python
n = len(nums)
# suf[i]表示nums[i+1:]的OR值
suf = [0]*n
for i in range(n-2,-1,-1):
    suf[i]=suf[i+1] | nums[i+1]
# pre表示nums[:i]的OR值
ans = pre = 0
for i in range(n):
    ans = max(ans,pre | (nums[i]<<k) | suf[i])
    pre |= nums[i]
return ans
```

二维（重点是一个顺序！！）

```python
n = len(grid)
m = len(grid[0])
MOD = 12345
ans = [[0]*m for _ in range(n)]

suf = 1# 表示从下一个元素到结尾的乘积
for i in range(n-1,-1,-1):
    for j in range(m-1,-1,-1):
        ans[i][j] = suf
        suf = suf*grid[i][j]%MOD

pre = 1
for i in range(n):
    for j in range(m):
        ans[i][j] = ans[i][j]*pre%MOD
        pre = pre*grid[i][j]%MOD
return ans
```

差分数组

```python
n = len(nums)
d = [0]*n
for l,r in queries:
    d[l]+=1
    d[r+1]-=1
a = [d[0]]
for i in range(1,n):
    a.append(a[i-1]+d[i])
```

## KMP算法

构建pi数组

```python
pi = [0] * n  # 前缀函数
for i in range(1, n):
    j = pi[i - 1]
    while j > 0 and s[i] != s[j]:
        j = pi[j - 1]
    if s[i] == s[j]:
        j += 1
    pi[i] = j
```

前缀中的周期

```python
for i in range(n):
    length = i + 1
    k = pi[i]
    if k > 0 and length % (length - k) == 0:
        print(length, length // (length - k))
```

字符串匹配

```python
next = build_next(pattern)
j = 0  # index for pattern

for i in range(len(text)):
    while j > 0 and text[i] != pattern[j]:
        j = next[j - 1]
    if text[i] == pattern[j]:
        j += 1
    if j == len(pattern):
        return i - j + 1  # Match found at this position
return -1  # No match
```

## Palindrome

在字符串前添加字符使其变为回文字符串

```python
rev_s = s[::-1]
combined = s + '#' + rev_s
next = build_kmp(combined) #kmp算法
prefix_len = next[-1]  # 最长回文前缀的长度
return rev_s[:len(s) - prefix_len] + s
```

插入最小字符使之变为回文

```python
n = len(s)
dp_prev = [0] * n
dp_curr = [0] * n

for i in range(n - 2, -1, -1):
    dp_curr[i] = 0  # 单字符为回文
    for j in range(i + 1, n):
        if s[i] == s[j]:
            dp_curr[j] = dp_prev[j - 1]
        else:
            dp_curr[j] = min(dp_prev[j], dp_curr[j - 1]) + 1
    dp_prev, dp_curr = dp_curr, [0] * n

return dp_prev[n - 1]
```

Manacher算法：最长回文子串

```python
def manacher(s):
    s = '#' + '#'.join(s) + '#'
    n = len(s)
    p = [0] * n
    c = r = 0
    for i in range(n):
        mirr = 2 * c - i
        if i < r:
            p[i] = min(r - i, p[mirr])
        while i + p[i] + 1 < n and i - p[i] - 1 >= 0 and s[i + p[i] + 1] == s[i - p[i] - 1]:
            p[i] += 1
        if i + p[i] > r:
            c, r = i, i + p[i]
    max_len, center_index = max((n, i) for i, n in enumerate(p))
    return s[center_index - max_len:center_index + max_len].replace('#', '')
```

### （7）质数

判断数量较少的数

```python
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```

判断数量很多的数

由于列表查找复杂度是O(n)，所以可以把结果保留在一个列表里然后用索引访问是否是质数，这样子会快一点

```python
#欧拉筛
def euler(r):
    prime = [0 for i in range(r+1)]
    common = []
    for i in range(2, r+1):
        if prime[i] == 0:
            common.append(i)
        for j in common:
            if i*j > r:
                break
            prime[i*j] = 1
            if i % j == 0:
                break
    return prime
#埃氏筛
def SieveOfEratosthenes(n, prime): 
    p = 2
    while (p * p <= n): 
        # If prime[p] is not changed, then it is a prime 
        if (prime[p] == True): 
            # Update all multiples of p 
            for i in range(p * 2, n + 1, p): 
                prime[i] = False
        p += 1

s = [True] * (10**2 + 1)
SieveOfEratosthenes(10**2, s)
print(s)
#建议自己先运行一遍看看0,1,2这种地方会不会出问题
```

## Sliding Window

统计完全子数组

```python
cnt = 0
set1 = set()
for num in nums:
    set1.add(num)
n = len(set1)
words = {}
wordsnum = 0
left = 0
for right in range(len(nums)):
    if nums[right] in words:
        words[nums[right]]+=1
    else:
        words[nums[right]]=1
        wordsnum+=1
    while wordsnum==n:
        cnt+=len(nums)-right
        words[nums[left]]-=1
        if words[nums[left]]==0:
            words.pop(nums[left])
            wordsnum-=1
        left+=1
return cnt
```

## Merge Sort

```python
def merge_sort(l):
    if len(l) <= 1:
        return l, 0
    mid = len(l) // 2
    left, left_count = merge_sort(l[:mid])
    right, right_count = merge_sort(l[mid:])
    l, merge_count = merge(left, right)
    return l, left_count + right_count + merge_count


def merge(left, right):
    merged = []
    left_index, right_index = 0, 0
    count = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] >= right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1
            count += len(left) - left_index
    merged += left[left_index:]+right[right_index:]
    return merged, count
```

### 区间问题

#### 区间合并：

给出一堆区间，要求**合并**所有**有交集的区间** （端点处相交也算有交集）。最后问合并之后的区间

【**步骤一**】：按照区间**左端点**从小到大排序。

【**步骤二**】：维护前面区间中最右边的端点为ed。从前往后枚举每一个区间，判断是否应该将当前区间视为新区间。

假设当前遍历到的区间为第i个区间 [li,ri]，有以下两种情况：

- li≤ed：说明当前区间与前面区间**有交集**。因此**不需要**增加区间个数，但需要设置 ed=max(ed,ri)。
- li>ed: 说明当前区间与前面**没有交集**。因此**需要**增加区间个数，并设置 ed=max(ed,ri)。

#### 选择不相交区间：

给出一堆区间，要求选择**尽量多**的区间，使得这些区间**互不相交**，求可选取的区间的**最大数量**。这里端点相同也算有重复。

【**步骤一**】：按照区间**右端点**从小到大排序。

【**步骤二**】：从前往后依次枚举每个区间。

假设当前遍历到的区间为第i个区间 [li,ri]，有以下两种情况：

- li≤ed：说明当前区间与前面区间有交集。因此直接跳过。
- li>ed: 说明当前区间与前面没有交集。因此选中当前区间，并设置 ed=ri。

#### 区间选点：

给出一堆区间，取**尽量少**的点，使得每个区间内**至少有一个点**（不同区间内含的点可以是同一个，位于区间端点上的点也算作区间内）。

转化为前一种问题

#### 区间覆盖问题：

给出一堆区间和一个目标区间，问最少选择多少区间可以**覆盖**掉题中给出的这段目标区间。

【**步骤一**】：按照区间左端点从小到大排序。

【**步骤二**】：**从前往后**依次枚举每个区间，在所有能覆盖当前目标区间起始位置start的区间之中，选择**右端点**最大的区间。

假设右端点最大的区间是第i个区间，右端点为ri。

最后将目标区间的start更新成ri。

#### 区间分组：

给出一堆区间，问最少可以将这些区间分成多少组使得每个组内的区间互不相交

【**步骤一**】：按照区间左端点从小到大排序。

【**步骤二**】：从**前往后**依次枚举每个区间，判断当前区间能否被放到某个现有组里面。

（即判断是否存在某个组的右端点在当前区间之中。如果可以，则不能放到这一组）

假设现在已经分了 m 组了，第 k 组最右边的一个点是 rk，当前区间的范围是 [Li,Ri] 。则：

如果$L_i \le r_k$ 则表示第 i 个区间无法放到第 k 组里面。反之，如果 Li>rk， 则表示可以放到第 k 组。

- 如果所有 m 个组里面没有组可以接收当前区间，则当前区间新开一个组，并把自己放进去。
- 如果存在可以接收当前区间的组 k，则将当前区间放进去，并更新当前组的 rk=Ri。

为了能快速的找到能够接收当前区间的组，我们可以使用**优先队列 （小顶堆）**。

优先队列里面记录每个组的右端点值，每次可以在 O(1) 的时间拿到右端点中的的最小值。

### 辅助栈

```python
stack1 = []
stack2 = []
def push(weight):
    stack1.append(weight)
    if not stack2:
        stack2.append(weight)
    elif weight<=stack2[-1]:
        stack2.append(weight)
def pop():
    if stack1:
        popout = stack1.pop(-1)
        if stack2[-1]==popout:
            popout2 = stack2.pop(-1)
def minweight():
    if stack2:
        print(stack2[-1])
while True:
    try:
        s = input()
    except EOFError:
        break
    s = s.split()
    if len(s)==1:
        if s[0]=='pop':
            pop()
        else:
            minweight()
    else:
        push(int(s[1]))
```

### 堆（优先队列）

```python
import sys
import heapq
from collections import defaultdict
input = sys.stdin.readline
 
minH = []
maxH = []
 
ldict = defaultdict(int)
rdict = defaultdict(int)
 
n = int(input())
 
for _ in range(n):
    op, l, r = map(str, input().strip().split())
    l, r = int(l), int(r)
    if op == "+":
        ldict[l] += 1
        rdict[r] += 1
        heapq.heappush(maxH, -l)
        heapq.heappush(minH, r)
    else:
        ldict[l] -= 1
        rdict[r] -= 1
    
    '''
    使用 while 循环，将最大堆 maxH 和最小堆 minH 中出现次数为 0 的边界移除。
    通过比较堆顶元素的出现次数，如果出现次数为 0，则通过 heappop 方法将其从堆中移除。
    '''
    while len(maxH) > 0 >= ldict[-maxH[0]]:
        heapq.heappop(maxH)
    while len(minH) > 0 >= rdict[minH[0]]:
        heapq.heappop(minH)
    #这实际上是一种“懒删除”策略，事实上对于堆你也只能这么删（汗）
    
    
    '''
    判断堆 maxH 和 minH 是否非空，并且最小堆 minH 的堆顶元素是否小于
    最大堆 maxH 的堆顶元素的相反数。
    '''
    if len(maxH) > 0 and len(minH) > 0 and minH[0] < -maxH[0]:
        print("Yes")
    else:
        print("No")
```

### 懒更新

```python
for i in range(n):
    c = arr[i][1]
    cnt[c] += 1
    if vis[c]:
        while cnt[Q[0][1]]: # 懒更新，每次只更新到堆中的最小值是实际的最小值
            f = heapq.heappop(Q)
            f = (f[0] + cnt[f[1]], f[1])
            heapq.heappush(Q, f)
            cnt[f[1]] = 0
```

### Greedy后悔

```python
import heapq


def max_potions(n, potions):
    # 当前健康值
    health = 0
    # 已经饮用的药水效果列表，用作最小堆
    consumed = []

    for potion in potions:
        # 尝试饮用当前药水
        health += potion
        heapq.heappush(consumed, potion)
        if health < 0:
            # 如果饮用后健康值为负，且堆中有元素
            if consumed:
                health -= consumed[0]
                heapq.heappop(consumed)


    return len(consumed)

n = int(input())
potions = list(map(int, input().split()))
print(max_potions(n, potions))
```

## 2.技巧

### （1）递归

```python
from functools import lru_cache
import sys
sys.setrecursionlimit(50000)
@lru_cache(maxsize=None)
def rec():
#如果有多个函数都需要缓存，需要将装饰器加在每个函数上
#不要加错位置！（这个很难debug）
```

### （2）enumrate

```python
nums = [5, 3, 9, 1, 6]
sorted_pairs = sorted(enumerate(nums), key=lambda x: x[1])
# 解包排序后的索引和值
sorted_indices, sorted_nums = zip(*sorted_pairs)
# 将两个列表打包成一个新的元组列表也可以使用 zip 
l1 = [1, 2, 3]
l2 = [4, 5, 6]
packed = zip(l1, l2)
```

### （3）bisect

```python
index = bisect.bisect(lst,number)#二分查找
```

### （5）一些有用的函数、技巧

```python
eval()#将字符串视为python表达式进行运算
print(f'{numbers} card(s)')#格式化输出
print(f'{number:.5f}')#保留5位小数
import math
math.ceil()#向正无穷取整
math.floor()#向负无穷取整
int()#去尾
round(digit)#四舍五入
s.lower()#将所有大写字母转化为小写字母，返回新字符串；类似的还有upper()
ord()#返回ascii
chr()#返回字符
s.find(sub)#查找一个子字符串在另一个字符串中的位置。如果找到了子字符串，则会返回子字符串的第一个字符在主字符串中的索引位置；如果没有找到子字符串，返回-1（只会返回第一个搜到的）
s.count(sub)#统计某个子串出现的次数

from math import gcd
print(gcd(a,b))#最大公约数内置函数
s.strip()#移除首尾的特定字符，默认是空白符，比如想移除'-'就s.strip('-')
#类似的还有lstrip移除开头的，rstrip移除结尾的

#在使用split()方法时，如果字符串中没有对应的分隔符，将返回一个只包含原字符串单一元素的列表；如果分隔符出现在字符串的首尾或者连续出现，那么 split() 方法将会把这些分隔符视为分隔点，并产生相应的空字符串项

#其它进制转十进制
decimal = int(string,digits) #string为其他进制字符串，digits为进制，比如2、4、8
#十进制转其他进制
binary = bin(num)
octal = oct(num)
hexadecimal = hex(num)
# 但要注意这样输出的是有前缀的，比如'0b1010'，所以要先切片

MIN=float('inf')
MAX=float('-inf')# 生成无限大无限小

#python的int自动满足大整数，甚至可以把’000123’这种字符串转成’123’
```

### （7）set

```python
union() intersection() difference()
#删除特定元素
set1.remove()
set1.discard()
#处理元素不存在的情况，remove()会报错，而discard()什么都不干
```

### （8）列表

```python
#python内置sort具有稳定性，如果想要对第一个元素排序，然后第一个元素相同情况下对第二个元素降序排序
#这时用reverse=True是不行的，因为reverse的底层逻辑是正常排一遍然后反过来
lst = [(1, 5), (2, 3), (1, 2), (2, 1), (1, 3)]
sorted_lst = sorted(lst, key=lambda x: (x[0], -x[1]))
#如果第二个元素为字符串不能取负，则需要用额外的方法
from functools import cmp_to_key
# 自定义比较函数
def compare_items(x, y):
    if x[0] < y[0]:
        return -1
    elif x[0] > y[0]:
        return 1
    elif x[1] < y[1]:  # 降序排序
        return 1
    elif x[1] > y[1]:
        return -1
    else:
        return 0

lst = [(1, 'z'), (2, 'c'), (1, 'a'), (2, 'b'), (1, 'y')]
sorted_lst = sorted(lst, key=cmp_to_key(compare_items))
#或者还可以分步进行，把需要倒序的放前面
lst.sort(key=lambda x: x[1], reverse=True)  # 按照x[1]降序排序
lst.sort(key=lambda x: x[0])             

l.remove(value)#移除列表中第一个匹配指定值的元素。如果列表中不存在该值，则会抛出一个 ValueError 异常
```

### （9）字典

```python
for key in my_dict:
for key in my_dict.keys():#遍历字典的键
for key in my_dict,values():#遍历字典的值
for key,value in my_dict.items():#遍历键值对
#以上的操作都是对一个视图对象进行遍历，并非是副本，因此它是可以实时改变字典内部内容的
if key in my_dict:#检测是否存在于字典的键中

#对字典进行排序
from collections import OrderedDict

my_dict = {'b': 2, 'c': 3, 'a': 1}
ordered_dict = OrderedDict(sorted(my_dict.items()))
print(ordered_dict)  # 输出: OrderedDict([('a', 1), ('b', 2), ('c', 3)])

#字典可以删除键值对O(1)
my_dict = {'a': 1, 'b': 2, 'c': 3}
del my_dict['b']
```

### （10）输入输出节约时间

```python
import sys
readin = sys.stdin.read().strip()
l = readin.split()
#一口气读进所有输入然后慢慢处理
#把答案都存在列表ans中
print('\n'.join(ans))
```

### （11）列表推导式

```python
#加判断的列表推导式
lnew = [str(x) for x in lold if x > n]
```

### （12）lambda函数

```python
l3 = list(map(lambda x, y: x + y, l1, l2))
# lambda x, y: x + y 是一个匿名函数，接受两个参数x和y，并返回它们的和
# map()函数会对l1和l2中的每个元素应用这个匿名函数
```

将一个比较函数转换为一个key函数

```python
from functools import cmp_to_key
def comparison(x):
    if x[-1]=='M':
        return float(x[:-1])*1
    else:
        return float(x[:-1])*1000
key_func = cmp_to_key(comparison)# 是否必要？存疑
sorted_list = sorted(my_list, key=key_func)
```

### （14）Counter

把一个列表中的重复数据组合起来，以{’key’:counted_times}的字典形式储存，时间复杂度O(n)，可以在后期减少很多遍历的耗时

```python
#统计众数的出现次数
from collections import Counter
print(max(Counter(input().split()).values()))
```

### （15）help

不少函数在math内置模块里有，如果考试的时候碰到记不得的可以用

```python
import math
help(math)
```

在终端慢慢找

### （16）product（笛卡尔积）

```python
from itertools import product
products = product(['0','1'],repeat=6)
for product in products:
    ...
```

### （17）permutation（排列）

```python
from itertools import permutations
permutation = permutations('ABCD',4)
for per in permutation:
    ...
    
import itertools
    permutations = list(itertools.permutations(nums))
    ans = [list(i) for i in permutations]
    return ans
```

## 其他注意点

图相关问题的行与列不要搞混

慎用字符串拼接做字典的键

在dfs中，特别特别要当心list的传递，养成传递副本的习惯，否则怎么也查不出错误

涉及到排序的题，注意检查reverse=True写了没，以及到底题目数据有没有排过序

注意数字的字符串形式和数值形式

用print调试完程序之后print不要忘记注释掉！！！

在很多时候python的运算精度其实是不够的，比如求三次根号n**(1/3)，你代入n=1000，它返回9.9999999998；所以能用乘法别用根号之类的运算

remove时间复杂度为O(n)，能不remove就留着，牺牲空间复杂度

有些题涉及非常大的数据，题目要求你对结果取模后输出，但实际上你往往得在中间运算中就取模，否则数据会爆掉

二维数组需要深度拷贝：import copy；A=copy.deepcopy(X)

有时候也可以考虑考虑brute force?比如一些保证有解的题目；或者说很难找到给出最值的方案，但是很容易判断一个值可不可以取到（这个时候可以用二分）（brute force与计算机思维）

python中比较数字和比较字符串的逻辑是不一样的，比如，11>9，'11'<'9'（逐位比较），看你需要哪一种

为了省事相似的代码块可以直接复制，但是但是！！！这种时候很容易变量改不干净！即使是简单题也容易WA

有些题目使用了lru_cache，视情况函数需要定义在循环体内部：比如Help Jimmy这道题，每一次循环都是一个全新的图，因此你必须重置你的lru_cache

输出是否要空行？是否除了ans之外要输出别的东西？

oj上的fstring中间不能有引号

```python
for i in range(1,len(nums)):
    if nums[i]>nums[0]:
        break
else:
    i=len(nums)# 这个else没写真的太难debug了
```

当存在多个函数时cache和调用不要写混了

如果presentation error考虑一下空格换行

有时候只处理前半部分输入太麻烦，可以考虑看一下后半部分的输入？根据要求的东西来处理数据可能可以减小复杂

Defaultdict：容易忘记删除：dic[a]-=1即使减完之后dic[a]==0也不会删掉这个键，即，a in dic返回True

很重要的错误来源是没有去重（一个数据重复出现，应该只算一次）
