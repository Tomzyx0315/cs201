k = 0
def printdir(prefixnum,dirname):
    prefixnum+=1
    print(prefix*prefixnum+dirname)
    files = []
    while True:
        s = input()
        if s == ']':
            files.sort()
            for i in range(len(files)):
                print(prefix*prefixnum+files[i])
            break
        if s[0] == 'd':
            printdir(prefixnum,s)
        else:
            files.append(s)
    return
while True:
    s = input()
    if s=='#':
        break
    if k!=0:
        print()
    k+=1
    print(f'DATA SET {k}:')
    print('ROOT')
    prefix = '|     '
    files = []
    if s[0]=='d':
        printdir(0,s)
    else:
        files.append(s)
    while True:
        s = input()
        if s == '*':
            files.sort()
            for i in range(len(files)):
                print(files[i])
            break
        if s[0]=='d':
            printdir(0,s)
        else:
            files.append(s)