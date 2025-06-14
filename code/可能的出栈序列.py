n = int(input())
def dfs(stack,output,index):
    if index==n+1:
        print(*(output+stack[::-1]))
        return
    if stack:
        dfs(stack[:-1],output+stack[-1],index)
        dfs(stack+str(index),output,index+1)
    else:
        dfs(stack+str(index),output,index+1)

dfs('','',1)