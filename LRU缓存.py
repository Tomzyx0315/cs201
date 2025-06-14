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


class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.num = 0
        self.hash = {}
        self.list = DoublyLinkedList()

    def get(self, key: int) -> int:
        if key in self.hash:
            value = self.hash[key].value
            self.list.remove(self.hash[key])
            newnode = ListNode(key,value)
            self.hash[key]=newnode
            self.list.append(newnode)
            return value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.hash:
            self.list.remove(self.hash[key])
            newnode = ListNode(key,value)
            self.hash[key]=newnode
            self.list.append(newnode)
        elif self.num<self.capacity:
            newnode = ListNode(key,value)
            self.hash[key]=newnode
            self.list.append(newnode)
            self.num+=1
        else:
            oldkey = self.list.pop()
            del self.hash[oldkey]
            newnode = ListNode(key,value)
            self.hash[key]=newnode
            self.list.append(newnode)
            

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)