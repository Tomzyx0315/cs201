# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True
        # 这个特判一定一定不要忘了！

        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        pre = None
        cur = slow
        while cur:
            next= cur.next
            cur.next = pre
            pre = cur
            cur = next
        
        left,right = head,pre
        while right:
            if left.val!=right.val:
                return False
            left = left.next
            right = right.next
        return True