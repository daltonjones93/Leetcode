# Index

1.  [Two Sum](#two-sum)
2.  [Valid Parentheses](#valid-parentheses)
3.  [Merge Two Sorted Lists](#merge-sorted-lists)
4.  [Best Time To Buy And Sell A Stock](#buy-sell-stock)


### [Two Sum](https://leetcode.com/problems/two-sum/)<a name="two-sum"></a>  

This is possibly the most classic leetcode problem. The punchline is that searching and and membership can be determined in O(1) time
by using a hash map. In a sense this uses a "hash" which maps keys to unique recoverable integers. This map can then be used to access
items.
```python
def twoSum(nums, target):

        num_dict = {} #use a dictionary (hash map) to be able to search for target-n in constant time

        for i, n in enumerate(nums):
            
            if target - n in num_dict: #this is an order (1) operation, determine if complement of n wrt target is in num_dict already
                return (num_dict[target-n], i) 

            num_dict[n] = i

```



### [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)<a name="valid-parentheses"></a>  

A classic use of a stack. Note that in this problem, we consider a sequence, where the sequence must hierarchically satisfy a condition.
This can be achieved by considering the last valid entry and the future entries via a stack. Also use the fact that popping and appending are order 1 operations. There are actually other ways to do this, potentially using two pointers or a while loop and replace,
but this is definitely the best way (most efficient).

```python
def isValid( s):
        """
        :type s: str
        :rtype: bool
        # """

        #this problem uses a stack! basically put elements on the top of a stack if
        #they dont violate the parentheses condition. Pop when parens complete.
        #return true if you end with empty stack

        stack = []
        left_b = set("([{")
        right_b_dict = {")":"(", "}":"{","]":"["}
        for l in s:
            if not stack and l in left_b:
                stack.append(l)
            elif stack and l in left_b:
                stack.append(l)
            elif not stack and l not in left_b:
                return False
            elif stack and l not in left_b:
                if right_b_dict[l] == stack[-1]:
                    stack.pop()
                else:
                    return False
        
        return len(stack) == 0
```

### [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)<a name="merge-sorted-lists"></a>  

I mean this is a technique that shows up all over the place. The operation can be done with O(1) memory and O(n) operations, you
just have to be a little bit careful about the details (for example, make sure you start with the right list and that you're
moving lists to the next spot once you add nodes. I didn't and you can get an infinite loop in a while loop, which is interesting,
but undesirable. This sort of thing should be second nature, so I should probably practice the pattern.

```python
def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """

        if list1 == None:
            return list2
        elif list2 == None:
            return list1
        
        if list1.val > list2.val:
            head = list2
            curr = head
            list2 = list2.next
        else:
            head = list1
            curr = head
            list1 = list1.next
        
        while list1 and list2:
            if list1.val <= list2.val:
                curr.next =list1
                list1 = list1.next
            else:
                curr.next =list2
                list2 = list2.next
            curr = curr.next
        
        if list1:
            curr.next = list1
            return head
        elif list2:
            curr.next = list2
            return head
        else:
            return head
```

### [Best Time To Buy And Sell A Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)<a name="buy-sell-stock"></a> 

You know, previously I had been solving this problem in a hacky way. I found a much more elegant way tracking the running minimum value and the computing profit afterward based on the difference between future values and that running minimum. Apparently this
is dynamic programming, but honestly, it doesn't really feel like there's a "subproblem" to me really. Maybe the two suproblems are 
what is the minimum up to the current point, and what would the resulting profit be if you bought and sold optimally in
this time period. So I suppose it is dynamic programming.

```python
def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        
        mx = 0
        mn_price = prices[0]
        for i in range(1,len(prices)):
            if prices[i] > mn_price:
                mx = max(mx,prices[i]-mn_price) #update max, this solves the subproblem
            elif prices[i] < mn_price:
                mn_price = prices[i] #update min for window
        
        return mx
```
