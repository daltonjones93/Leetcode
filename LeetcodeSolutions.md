# Index

1.  [Two Sum](#two-sum)
2.  [Valid Parentheses](#valid-parentheses)
3.  [Merge Two Sorted Lists](#merge-sorted-lists)
4.  [Best Time To Buy And Sell A Stock](#buy-sell-stock)
5.  [Valid Palindrome](#valid-palindrome)
6.  [Invert Binary Tree](#invert-binary-tree)
7.  [Valid Anagram](#valid-anagram)
8.  [Binary Search](#binary-search)
9.  [Flood Fill](#flood-fill)
10. [Lowest Common Ancestor](#lowest-common-ancestor)
11. [Balanced Binary Tree](#balanced-binary-tree)
12. [Has Cycle](#has-cycle)
13. [Implement Queue Using Stacks](#implement-queue-using-stacks)
14. [Ransom Note](#ransom-note)
15. [First Bad Version](#first-bad-version)
16. [Climbing Stairs](#climbing-stairs)
17. [Longest Palindrome](#longest-palindrome)
18. [Reverse Linked List](#reverse-linked-list)
19. [Majority Element](#majority-element)
20. [Add Binary](#add-binary)
21. [Diameter of Binary Tree](#diameter-of-binary-tree)
22. [Middle of the Linked List](#middle-of-the-linked-list)
23. [Contains Duplicate](#contains-duplicate)
24. [Roman To Integer](#roman-to-integer)
25. [Backspace Compare](#backspace-compare)
    


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

### [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)<a name="valid-palindrome"></a> 

This question is a good example of using two pointers to check if something is the same front and back. The tricky parts are actually reading the problem to determine what actually constitutes a palindrome (only compare lowercase alphanumeric characters.) Nested while loops seem to do this effectively, and we get an order(n) solution.

```python
def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """

        letter_set = set(string.ascii_lowercase + "0123456789")
        s = s.lower()

        l = 0
        r = len(s) - 1
        while l < r:
            while l < r and s[l] not in letter_set:
                l += 1
            while l < r and s[r] not in letter_set:
                r -= 1
            if l == r:
                break
            if s[r] != s[l]:
                return False
            l += 1
            r -= 1

        return True

```

### [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)<a name="invert-binary-tree"></a>  

This is a good example of using recursive functions. Essentially, set up the base case (end of the recursion) then perform the operation (this could come before or after recursive call, in this case before) then call the recursion. In this case, we switch the left and right nodes of the current root, then pass to left and right nodes and do the same thing there. 

```python
def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root == None:
            return


        root.right, root.left = root.left, root.right
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root


```

### [Valid Anagram](https://leetcode.com/problems/valid-anagram/)<a name="valid anagram"></a>  

I suppose the lesson here is that when counting (and particularly when order doesn't matter, use a dictionary.) In our case we use the collections.Counter class since it's constructor automatically counts things

```python
def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """

        if len(s) != len(t):
            return False

        return collections.Counter(s) == collections.Counter(t)
```

### [Binary Search](https://leetcode.com/problems/binary-search/)<a name="binary search"></a> 
```python
def search(self, nums: List[int], target: int) -> int:
        
        #arguably binary search is maybe one of the most 
        #powerful ideas in cs. Basically you can cut a list in 
        #half every time you look for an entry. Since it's sorted
        #you know where to look, thus if you divide it by 2^n times,
        #you get to one after len(nums)/2^n = 1 when n = log2(len(nums))
        #much better than linear.
        
        #also important to point out, you can do it recursively, but this
        #involves more function calls and is slower. Better to convert
        #the "tail recursion" into a while loop
        
        l = 0
        r = len(nums)-1
        
        while l < r:
            m = (l+r) //2 #Note, this rounds down
            if nums[m] == target:
                return m #terminate early
            elif nums[m] < target:
                l = m+1 #number must be in a position greater than m
            else:
                r = m-1 #number must be in a position less than m
        
        if nums[l] == target:
            return l
        
        return -1

```
### [Flood Fill](https://leetcode.com/problems/flood-fill/)<a name="flood fill"></a> 

This is a great problem to practice both depth first and breadth first searches (as well as practicing the leetcode conventions of breaking recursion when you leave the edge of a board, image etc.) I've included implementations of the bfs and the dfs solutions here. Both are O(n).

```python
def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:

        #I guess there's lots of ways to do this, just for variety,
        #let's do the ol bfs

        #bfs
        if image[sr][sc] == color:
            return image
        q = collections.deque([(sr,sc)])
        dirs = [0,1,0,-1,0]
        orig_color = image[sr][sc]
        while q:
            x,y = q.popleft()
            image[x][y] = color
            for i in range(4):
                nx = x + dirs[i]
                ny = y + dirs[i+1]
                if nx >= 0 and ny >= 0 and nx < len(image) and ny < len(image[0]) and image[nx][ny] == orig_color:
                    q.append((nx,ny))
        
        return image


def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        if image[sr][sc] == color:
            return image
        self.dirs = [0,1,0,-1,0]
        self.orig_color = image[sr][sc]

        def dfs(x,y):

            image[x][y] = color
            for i in range(4):
                nx = x + self.dirs[i]
                ny = y + self.dirs[i+1]
                if nx >= 0 and ny >= 0 and nx < len(image) and ny < len(image[0]) and image[nx][ny] == self.orig_color:
                    dfs(nx,ny)
        
        dfs(sr,sc)
        return image

```

### [Lowest Common Ancestor](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)<a name="lowest common ancestor"></a> 
```python
def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        
        #the point of a binary search tree is that elements to the left of a node
        #are less than a node, and greater on the right. This solution essentially becomes a binary
        #search

        if p.val > q.val:
            return self.lowestCommonAncestor(root,q,p)

        if p.val <= root.val <= q.val:
            return root
        elif q.val < root.val:
            #move left
            return self.lowestCommonAncestor(root.left,p,q)
        else:
            return self.lowestCommonAncestor(root.right,p,q)
```

### [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)<a name="balanced binary tree"></a> 

Plenty of ways to solve this problem, in particular, we need to solve for the height of each node, then measure the height of each 
left and right branch.

```python

def height(self,root):

        if not root:
            return 0

        return 1 + max(self.height(root.left),self.height(root.right))


def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        h1 = self.height(root.left)
        h2 = self.height(root.right)

        if abs(h1-h2) > 1:
            return False
        
        return self.isBalanced(root.left) and self.isBalanced(root.right)
        

```
### [Has Cycle](https://leetcode.com/problems/has-cycle/)<a name="has cycle"></a> 

The trick here is to use one fast node and one slow node. This technique shows up in other places as well (finding middle of the list for example.) But yeah cycle detection should probably be done this way. Several things to node. Start fast and slow at different spots, and only put while loop conditions on fast (since it's going to reach the end of the list first, since it's fast.)

```python
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """

        #this is a classic algorithm. 

        #classic solution:
        slow = head
        fast = head.next
        while fast and fast.next:
            if slow == fast:
                return True
            slow = slow.next
            fast = fast.next.next
            
        return False
```

### [Implement Queue Using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)<a name="implement queue using stacks"></a> 

Turns out I did not read the instructions here, but below we'll find the correct code. The idea is to, when you need to access 
the first element of the input list, keep popping it and appending the result to the output, which should be in the reverse order. Only do this when output is empty, since otherwise it would mess up the order. 

```python
class MyQueue(object):

    def __init__(self):
        
        # self.L = []
        
        # self.Q = []
        # self.lind = 0

        self.input = []
        self.output = []
        

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        # self.L.append(x)
        self.input.append(x)
        

    def pop(self):
        """
        :rtype: int
        """

        #for this we need to pop the back of the list input
        if self.output:
            return self.output.pop()
        else:
            while self.input:
                self.output.append(self.input.pop())
        
        return self.output.pop()
        # return self.L.pop(0)
        # v = self.Q[self.lind]
        # self.lind += 1
        # return v
        

    def peek(self):
        """
        :rtype: int
        """
        # return self.L[0]
        if self.output:
            return self.output[-1]
        else:
            while self.input:
                self.output.append(self.input.pop())
        
        return self.output[-1]
        

        

    def empty(self):
        """
        :rtype: bool
        """
        # return len(self.L) == 0

        return (not self.input) and (not self.output)
```


### [Ransom Note](https://leetcode.com/problems/ransom-note/)<a name="ransom note"></a> 
The trick here is to note that order doesn't matter, we just need to track letter frequency. The collections.Counter module
is very convenient for this (similar to defaultdict it doesn't throw key errors.) 

```python
def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """

        #another counting problem. Note that order
        #doesnt matter! which should make us think of a dictionary
        #or a set
        cm = collections.Counter(magazine)
        cr = collections.Counter(ransomNote)
        for l in cr:
            if cm[l] < cr[l]:
                return False
        return True
```

### [First Bad Version](https://leetcode.com/problems/first-bad-version/)<a name="first bad version"></a> 
The trick here is recognizing we can use binary search really, the rest is pretty simple.


```python
class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """

        #the trick here is to recognize that we can perform binary search
        #instead of the condition relating to < target etc we have the version
        #condition, which amounts to the same thing

        l = 1
        r = n
        
        while l < r:
            m = (l + r) // 2

            b = isBadVersion(m)
            if b:
                r = m #the first bad version must be in the interval l,m
            else:
                l = m + 1 #the first bad version must be in the interval m +1, l
            
        return l
```

### [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)<a name="climbing stairs"></a> 
Ah, the first foray into dynamic programming. The paradigm is this (with some variation): If I know the answer to the smaller problem can I solve the larger problem? Do I know the answer to the smallest problem? Then I can build a solution efficiently. In this case, we know there are only two ways to get to step k (from step k-1, k-2) hence the number of ways to get to step k is the number of ways to get to step k-1 plus the number of ways to get to step k-2. 

```python
def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """

        #how many ways can you get to stair n? really just two
        #one step from n-1 and one step from step n-2. We just need 
        #to iterate on that

        if n == 1:
            return 1
        if n == 2:
            return 2

        prev = 1
        curr = 2
        for i in range(3,n+1):
            nxt = curr + prev
            prev = curr
            curr = nxt
            
        return curr

```


### [Longest Palindrome](https://leetcode.com/problems/longest-palindrome/)<a name="longest palindrome"></a>  
This question always trips me up. Good lesson to actually read and understand the problem before working on it. It's not asking what the largest palindrome contained with s is (I think this requires a O(n^2) solution) it's asking what's the biggest palindrome can you make out of the letters in s. I think it's deliberately misleading.

```python
def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """

        c = collections.Counter(s)
        odd_found = False
        pal_size = 0
        for k in c:
            if c[k] % 2 ==1:
                odd_found = True
            pal_size += (c[k]//2)*2
        if odd_found:
            pal_size += 1
        return pal_size

```

### [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)<a name="reverse linked list"></a>  

```python
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        
        #oh man, this gets used seriously all the time. This and 
        #binary search everyone should be able to do in their sleep.

        #edge cases:
        if not head:
            return head
        if not head.next:
            return head

        
        prev = head
        curr = head.next
        prev.next = None #this is critical

        while curr:
            tmp = curr.next
            curr.next = prev
            prev = curr
            curr = tmp
        
        return prev
```

### [Majority Element](https://leetcode.com/problems/majority-element/)<a name="majority element"></a> 
```python
def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        #several ways to do this, all O(n).

        #this first method is somehow more efficient memory wise (it really shouldn't be.)
        # c = collections.Counter(nums)
        # for k in c:
        #     if c[k] > len(nums) // 2:
        #         return k

        #second method, we know that the majority element must occur greater
        #than half the time, meaning that if we keep a running count of the majority
        #element so far, eventually the majority element will dominate.

        #second method is O(1) memory use, O(n) time
        m = nums[0]
        cnt = 1
        for i in range(1,len(nums)):
            if nums[i] == m:
                cnt += 1
            else:
                cnt -= 1
            
            if cnt == 0:
                m = nums[i]
                cnt = 1
        
        return m
```

### [Add Binary](https://leetcode.com/problems/add-binary/)<a name="add binary"></a> 

```python
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """

        #one of those cute one liners...

        return "{0:b}".format(int(a,2) + int(b,2))

        #really the spirit of the question I think would be to 
        #convert the binary number to a base 10 number by hand
        #or directly add the strings together. Which sounds annoying.
        #could also use bin function
```

### [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)<a name="diameter of binary tree"></a> 

I love this question so much. So many great things to practice here, between designing a recursive function to return information relevant to higher recursive calls, working on edge cases and base cases for recursion and maximizing a related function within the recursive function. Great.
```python
class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        #god I love trees. Great practice for recursive functions.

        #what we want is a function that can calculate the height of each 
        #node and track the maximum diameter containing each node,
        if root == None:
            return root
        
        self.diam = 0
        def get_diam(root):

            if root == None:
                return 0
            
            lh = get_diam(root.left)
            rh = get_diam(root.right)
            self.diam = max(self.diam, lh + rh)
            return 1 + max(lh,rh)
        
        get_diam(root)
        return self.diam
```


### [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)<a name="middle of the linked list"></a> 

This is another very useful trick that used in a fair amount of other problems. So we can use fast and slow nodes to find specific locations within a list and to detect cycles. And probably other things too, but those are the applications we've seen for this method so far.
```python
def middleNode(self, head):
        """
        # :type head: ListNode
        # :rtype: ListNode
        # 
        """

        #Another fast slow sort of thing. 
        #you could also count to the end of the list and go back, which actually I beleive is the same number of operations with less memory? (Although they're both 
        #O(1))

        #solution 1
        cnt = 0
        curr = head
        while curr:
            curr = curr.next
            cnt+=1
        
        mid = cnt // 2 #think about why this is true!! Indexing is very annoying
        cnt = 0
        curr = head
        while curr:
            if cnt == mid:
                return curr
            curr = curr.next
            cnt+=1

        #solution 2: This is really what they're looking for 
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        #think about why this always results in the right answer (small cases suffice think 3/4 nodes)

        return slow

```

### [Maximum Depth](https://leetcode.com/problems/maximum-depth/)<a name="maximum depth"></a> 
I guess computing the height of a branch is an important pattern for later problems, whether we're computing the height, or say summing nodes or finding the maximum value along a path or something. All the same pattern.
```python
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        #isn't this just computing the height??
        if root == None:
            return 0
        
        return 1 + max(self.maxDepth(root.left),self.maxDepth(root.right))
```

### [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)<a name="contains duplicate"></a> 
Another example of the usefulness of a set (hashmap) to keep track of the number of things.
```python
def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        #just use a set

        # return len(set(nums)) != len(nums)

        #memory isn't so good

        c = collections.Counter()
        while nums:
            n = nums.pop()
            if c[n] > 0:
                return True
            c[n] += 1
        return False

```



### [Roman To Integer](https://leetcode.com/problems/roman-to-integer/)<a name="roman to integer"></a> 
This problem has a sort of stack feel in that we're traversing a string backwards (we could actually
convert it into a list and pop off elements over time too, we just don't need to. Numerals have different meanings based on their ordering, so just keep track of s[i], s[i+1] as i moves backwards and you can perform the op. 
```python
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """

        d = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500, "M":1000}

        n = 0
        for i in reversed(range(len(s))):

            if i != len(s)-1 and d[s[i]] < d[s[i+1]]:
                n -= d[s[i]]
            
            else:
                n+= d[s[i]]

        return n

```


### [Backspace Compare](https://leetcode.com/problems/backspace-compare/)<a name="backspace compare"></a> 
Man I originally thought this problem was impossible (which, if you don't use a stack, it is super annoying.) Punchline? Use a stack.
```python
def backspaceCompare(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """


        #I think I was making this way too hard before. This is pretty clearly
        #just a stack problem. I've come a long way since march.

        sstack =[]
        tstack = []

        for l in s:
            if l == "#":
                if sstack:
                    sstack.pop()
            else:
                sstack.append(l)
        for l in t:
            if l == "#":
                if tstack:
                    tstack.pop()
            else:
                tstack.append(l)
        
        return sstack == tstack

```





