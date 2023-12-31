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
26. [Count Bits](#count-bits)
27. [Same Tree](#same-tree)
28. [Number of 1 Bits](#number-of-1-bits)
29. [Longest Common Prefix](#longest-common-prefix)
30. [Single Number](#single-number)
31. [Palindrome Linked List](#palindrome-linked-list)
32. [Move Zeros](#move-zeros)
33. [Symmetric Tree](#symmetric-tree)
34. [Missing Number](#missing-number)
35. [Palindrome Number](#palindrome-number)
36. [Convert Sorted Array To Binary Search Tree](#convert-sorted-array-to-binary-search-tree)
37. [Reverse Bits](#reverse-bits)
38. [Subtree of Another Tree](#subtree-of-another-tree)
39. [Squares of A Sorted Array](#squares-of-a-sorted-array)
40. [Maximum Subarray](#maximum-subarray)
41. [Update Matrix](#update-matrix)
42. [K Closest Points To Origin](#k-closest-points-to-origin)
43. [Longest Substring Without Repeating Characters](#Longest-Substring-Without-Repeating-Characters)
44. [3Sum](#3sum)
45. [Course Schedule](#course-schedule)
46. [Evaluate Reverse Polish Notation](#evaluate-reverse-polish-notation)
47. [Implement Trie](#implement-trie)
48. [Coin Change](#coin-change)
49. [Validate Binary Tree](#validate-binary-search-tree)
50. [Number of Islands](#number-of-islands)
51. [Rotting Oranges](#rotting-oranges)
52. [Product of Array Except Self](#product-of-array-except-self)
53. [Min Stack](#min-stack)
54. [Search In Rotated Array](#search-in-rotated-array)
55. [Permutations](#permutations)
56. [Combination Sum](#combination-sum)
57. [Lowest Common Ancestor of a Binary Tree](#lowest-common-ancestor-of-a-binary-tree)
58. [Merge Intervals](#merge-intervals)
59. [Time Based Key-Value Store](#time-based-key-value-store)
60. [Accounts Merge](#accounts-merge)
61. [Spiral Matrix](#spiral-matrix)
62. [String To Integer Atoi](#string-to-integer-atoi)
63. [Partition Equal Subset Sum](#partition-equal-subset-sum)
64. [Word Break](#word-break)
65. [Subsets](#subsets)
66. [Container With Most Water](#container-with-most-water)
67. [Unique Paths](#unique-paths)
68. [Binary Tree Right Side View](#binary-tree-right-side-view)
69. [Longest Palindromic Substring](#longest-palindromic-substring)
70. [Construct Binary Tree From Preorder and Inorder Traversal](#construct-binary-tree-from-preorder-and-inorder-traversal)
71. [Letter Combinations of a Phone Number](#letter-combinations-of-a-phone-number)
72. [Word Search](#word-search)
73. [Find Anagrams](#find-anagrams)
74. [Daily Temperatures](#daily-temperatures)
75. [Kth Smallest Element in a Binary Search Tree](#kth-smallest-element-in-a-binary-search-tree)
76. [House Robber](#house-robber)
77. [Task Scheduler](#task-scheduler)
78. [Minimum Height Trees](#minimum-height-trees)
79. [Gas Station](#gas-station)
80. [Valid Sudoku](#valid-sudoku)
81. [Next Permutation](#next-permutation)
82. [Maximum Product Subarray](#maximum-product-subarray)
83. [Remove Nth Node From End of List](#remove-nth-node-from-end-of-list)
84. [Design Add and Search Words Data Structure](#design-add-and-search-words-data-structure)
85. [Pacific Atlantic Water Flow](#pacific-atlantic-water-flow)
86. [Group Anagrams](#group-anagrams)
87. [Swap Nodes in Pairs](#swap-nodes-in-pairs)
88. 

    

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

### [Valid Anagram](https://leetcode.com/problems/valid-anagram/)<a name="valid-anagram"></a>  

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

### [Binary Search](https://leetcode.com/problems/binary-search/)<a name="binary-search"></a> 
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
### [Flood Fill](https://leetcode.com/problems/flood-fill/)<a name="flood-fill"></a> 

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

### [Lowest Common Ancestor](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)<a name="lowest-common-ancestor"></a> 
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

### [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)<a name="balanced-binary-tree"></a> 

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


### [Ransom Note](https://leetcode.com/problems/ransom-note/)<a name="ransom-note"></a> 
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

### [First Bad Version](https://leetcode.com/problems/first-bad-version/)<a name="first-bad-version"></a> 
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

### [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)<a name="climbing-stairs"></a> 
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


### [Longest Palindrome](https://leetcode.com/problems/longest-palindrome/)<a name="longest-palindrome"></a>  
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

### [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)<a name="reverse-linked-list"></a>  

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

### [Majority Element](https://leetcode.com/problems/majority-element/)<a name="majority-element"></a> 
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

### [Add Binary](https://leetcode.com/problems/add-binary/)<a name="add-binary"></a> 

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

### [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)<a name="diameter-of-binary-tree"></a> 

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


### [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)<a name="middle-of-the-linked-list"></a> 

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

### [Maximum Depth](https://leetcode.com/problems/maximum-depth/)<a name="maximum-depth"></a> 
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

### [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)<a name="contains-duplicate"></a> 
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



### [Roman To Integer](https://leetcode.com/problems/roman-to-integer/)<a name="roman-to-integer"></a> 
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


### [Backspace Compare](https://leetcode.com/problems/backspace-compare/)<a name="backspace-compare"></a> 
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

### [Count Bits](https://leetcode.com/problems/count-bits/)<a name="count-bits"></a> 

```python
def countBits(self, n):
        """
        :type n: int
        :rtype: List[int]
        """

        if n == 0:
            return [0]
        if n == 1:
            return [0,1]
        if n == 2:
            return [0,1,1]
        dp = [0,1,1]

        for i in range(3,n+1):
            dp.append(dp[i >> 1] + i % 2)
        
        return dp
```
### [Same Tree](https://leetcode.com/problems/same-tree/)<a name="same-tree"></a> 
```python
def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """

        #recurse together...
        if not p and not q:
            return True
        elif not p and q:
            return False
        elif not q and p:
            return False
        elif p.val != q.val:
            return False
        
        
        return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)
```

### [Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/)<a name="number-of-1-bits"></a> 
```python
def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """

        return bin(n).count("1")



        # h_weight = 0
        # while n > 0:
        #     h_weight += n % 2
        #     n >>= 1
        # return h_weight
```


### [Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/)<a name="longest-common-prefix"></a> 

```python
def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """

        

        pr = ""
        for i in range(len(strs[0])):
            cand = strs[0][i]
            for s in strs:
                if i >= len(s) or s[i] != cand:
                    return pr
            pr += cand
        
        return pr
```
### [Single Number](https://leetcode.com/problems/single-number/)<a name="single-number"></a> 
```python
def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        snums = collections.Counter()
        while nums:
            n = nums.pop()
            if snums[n] >0:
                del snums[n]
            else:
                snums[n] =1
        
        return snums.keys()[0]

```
### [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)<a name="palindrome-linked-list"></a> 
```python
def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """

        #ah, a beautiful combination of two other problems we've done

        #first, find the middle of the list, 
        #second reverse the list
        #third traverse the reversed list and the forward half at the same time
        #to make sure they're equal.

        #the only trick is how far to reverse and how far to check...

        if not head.next:
            return True
        if not head.next.next:
            if head.val == head.next.val:
                return True
            else:
                return False

        
        
        #find middle:
        
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        if fast: #this means it's odd
            slow = slow.next
        
        #reverse list
        prev = slow
        curr = slow.next
        prev.next = None

        while curr:
            tmp = curr.next
            curr.next = prev
            prev = curr
            curr = tmp
        
        #prev is the new head
        while prev:
            if prev.val != head.val:
                return False
            prev = prev.next
            head = head.next
        
        return True
```
### [Move Zeros](https://leetcode.com/problems/move-zeros/)<a name="move-zeros"></a> 
```python
def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """


        #this is really just a question of keeping track of the number of zeros
        #and then just translating back
        nzero = 0
        for i in range(len(nums)):
            if nums[i]== 0:
                nzero+=1
            else:
                nums[i-nzero] = nums[i]
        
        for i in range(1, nzero+1):
            nums[-i] = 0
        return nums
```

### [Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)<a name="symmetric tree"></a> 
This one definitely gave me fits. Check pairs at a time, not their children.
``` python
def isSymmetric(self, root: Optional[TreeNode]) -> bool:

        def rec_check(r1,r2):
            
            if r1 == r2 == None:
                return True
            elif r1 == None or r2 == None:
                return False
            elif r1.val != r2.val:
                return False

            return rec_check(r1.left,r2.right) and rec_check(r1.right,r2.left)
        
        return rec_check(root.left,root.right)
```

### [Missing Number](https://leetcode.com/problems/missing-number/)<a name="missing-number"></a> 
Credit to ol Gauss for this one. Jesus, what a genius.
```python
def missingNumber(self, nums: List[int]) -> int:

        return (len(nums) * (len(nums)+1))//2 - sum(nums)
```

### [Palindrome Number](https://leetcode.com/problems/palindrome-number/)<a name="palindrome-number"></a>
Pretty straightforward, use two pointers, convert to a string.
```python
def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        
        
        x = str(x)
        l = 0
        r = len(x) - 1
        while l < r:
            if x[l] != x[r]:
                return False
            l +=1
            r -=1
        return True
```

### [Convert Sorted Array To Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)<a name="convert-sorted-array-to-binary-search-tree"></a>  
Really Can't say enough about this problem. Such a good example of how to think about recursion, binary search, base/edge cases, you name it. 
```python
def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """


        def build_tree(l,r):
            if l > r:
                return None
            m = (l+r)//2
            root = TreeNode(nums[m])
            root.left = build_tree(l,m-1)
            root.right = build_tree(m+1, r)
            return root

        return build_tree(0,len(nums)-1)
```

### [Reverse Bits](https://leetcode.com/problems/reverse-bits/)<a name="reverse-bits"></a>
```python
def reverseBits(self, n: int) -> int:

         
        b = '{:032b}'.format(n)
        b = b[::-1]
        return int(b,2)
```


### [Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/)<a name="subtree-of-another-tree"></a>

```python
def isEqual(root, subRoot):

    if not root and not subRoot:
        return True
    elif not root or not subRoot:
        return False
    elif root.val != subRoot.val:
        return False
    
    return isEqual(root.left,subRoot.left) and isEqual(root.right, subRoot.right)

class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:

        if isEqual(root,subRoot):
            return True
        
        if root.left and root.right:
            return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
        elif root.left:
            return self.isSubtree(root.left, subRoot)
        elif root.right:
            return self.isSubtree(root.right, subRoot)
        else:
            return False
```


### [Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/)<a name="squares-of-a-sorted-array"></a>
```python
def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        neg = False
        pnums = []
        nnums = collections.deque()
        for n in nums:
            if n < 0:
                nnums.appendleft(n**2)
            else:
                pnums.append(n**2)
        
        if not nnums:
            return pnums
        elif not pnums:
            return nnums

        del nums
        
        ret = []
        indn = 0
        indp = 0

        while indn < len(nnums) and indp < len(pnums):
            if nnums[indn] < pnums[indp]:
                ret.append(nnums[indn])
                indn += 1
            else:
                ret.append(pnums[indp])
                indp += 1
        if indn < len(nnums):
            
            ret += list(collections.deque(itertools.islice(nnums, indn, len(nnums))))
        elif indp < len(pnums):
            ret += pnums[indp:]
        return ret

```


### [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)<a name="maximum-subarray"></a>  

This is such an important pattern and problem. Well worth thinking about why the below solution works (dp really.)

```python

def maxSubArray(self, nums: List[int]) -> int:

        
        m = nums[0]
        p = nums[0]
        for i in range(1,len(nums)):
            p = max(nums[i],nums[i] + p) #question, do we include the previous number in the current subarray? prior max interval up to i-1 containing i-1 is p
            m = max(m,p)

        return m

```

### [Insert Interval](https://leetcode.com/problems/insert-interval/)<a name="insert-interval"></a>  

This is such a rough question in terms of corner cases, definitely worth doing it multiple times in the future.

```python
def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:

        #looking at this again, this feels like a stack question.

        if not intervals:
            return [newInterval]

        #some corner cases
        if newInterval[1] < intervals[0][0]:
            return [newInterval]+intervals
        elif newInterval[0] > intervals[-1][1]:
            return intervals+[newInterval]
        
        stack = []
        i = 0
        while i < len(intervals):
            ivl = intervals[i]
            if ivl[1] < newInterval[0]:
                stack.append(ivl)
                i += 1
            elif ivl[0] > newInterval[1]:
                stack.append(ivl)
                i += 1
            else:
                stack.append([min(ivl[0],newInterval[0]),max(ivl[1],newInterval[1])])
                while i < len(intervals) and intervals[i][0] <= stack[-1][1]:
                    stack[-1][1] = max(stack[-1][1],intervals[i][1])
                    i += 1

            if i < len(intervals) and intervals[i-1][1] < newInterval[0] and newInterval[1] < intervals[i][0]:
                stack.append(newInterval)
                    

        return stack

```


### [Update Matrix](https://leetcode.com/problems/update-matrix/)<a name="update-matrix"></a>  
```python
def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:

        q = collections.deque()
        visited = set()
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if mat[i][j] == 0:
                    q.append((i,j))
                    visited.add((i,j))
        dirs = [0,1,0,-1,0]
        while q:
            x,y = q.popleft()

            for i in range(4):
                nx,ny = x+ dirs[i],y + dirs[i+1]
                if nx >= 0 and ny >= 0 and nx < len(mat) and ny < len(mat[0]) and (nx,ny) not in visited:
                    visited.add((nx,ny))
                    q.append((nx,ny))
                    mat[nx][ny] = 1 + mat[x][y]
        
        return mat
```




### [K Closest Points To Origin](https://leetcode.com/problems/k-closest-points-to-origin/)<a name="k-closest-points-to-origin"></a>  
I love a good python 1 liner.
```python
def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:

        return sorted(points, key = lambda x:x[0]**2 + x[1]**2)[:k]
```

### [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)<a name="longest-substring-without-repeating-characters"></a>  
```python
def lengthOfLongestSubstring(self, s: str) -> int:

        #think sliding window
        if s == "":
            return 0

        l = 0
        r = 1
        c = defaultdict(int)
        c[s[0]] = 1
        m = 1
        while r < len(s):

            c[s[r]] += 1

            while c[s[r]] > 1:
                c[s[l]]-= 1
                l += 1

            m = max(r-l+1,m)

            r +=1
        return m
```


### [3sum](https://leetcode.com/problems/3sum/)<a name="3sum"></a>  
```python
def threeSum(self, nums: List[int]) -> List[List[int]]:

        nums.sort()
        ret = []
        for i in range(len(nums)-2):
            l = i+1
            r = len(nums)-1
            while l < r:
                if nums[i] + nums[l] + nums[r] > 0:
                    r -= 1
                elif nums[i] + nums[l] + nums[r] < 0:
                    l += 1
                else:
                    ret.add((nums[i],nums[l],nums[r]))
                    l += 1
                    r -= 1
        return ret
```

### [Course Schedule](https://leetcode.com/problems/course-schedule/)<a name="course-schedule"></a>  

```python
def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:

        #we just need to find a cycle if one exists
        if len(prerequisites) == 0:
            return True

        graph = defaultdict(list)
        for x,y in prerequisites:
            graph[y].append(x)
        
        #need to find a cycle if it exists

        states = [0]*numCourses

        def dfs(x):

            for nbr in graph[x]:
                if states[nbr] == 0:
                    states[nbr] = 1
                    if not dfs(nbr):
                        return False
                    states[nbr] = 2
                
                elif states[nbr] == 1:
                    return False
            
            return True

        for i in range(numCourses):
            if states[i] == 0:
                states[i] = 1
                if not dfs(i):
                    return False
                states[i] = 2
        return True

```

### [Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)<a name="evaluate-reverse-polish-notation"></a>  

```python
def evalRPN(self, tokens: List[str]) -> int:

        ops = "+/-*"
        
        stack = []
        for l in tokens:
            if l not in ops:
                stack.append(int(l))
            else:
                n1 = stack.pop()
                n2 = stack.pop()
                if l == "+":
                    stack.append(n1 + n2)
                elif l == "-":
                    stack.append(n2-n1)
                elif l == "*":
                    stack.append(n2*n1)
                elif l == "/":
                    stack.append(int(n2/n1))
        
        return stack[0]

```

### [Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree/)<a name="implement-trie"></a>  
Great one to practice, very interesting data structure. 
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for l in word:
            if l not in node.children:
                node.children[l] = TrieNode()
            node = node.children[l]
        
        node.word = True
        

    def search(self, word: str) -> bool:
        node = self.root
        for l in word:
            if l not in node.children:
                return False
            node = node.children[l]
        return node.word

        

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for l in prefix:
            if l not in node.children:
                return False
            node = node.children[l]
        return True
```

### [Coin Change](https://leetcode.com/problems/coin-change/)<a name="coin-change"></a> 

```python

def coinChange(self, coins: List[int], amount: int) -> int:

        dp = [math.inf]*(amount + 1)
        dp[0] = 0
        for c in coins:
            for i in range(c,len(dp)):
                dp[i] = min(1+dp[i-c],dp[i])
                
        if dp[-1] == math.inf:
            return -1
        
        return dp[-1]

```



### [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)<a name="validate-binary-search-tree"></a>  
```python
def isValidBST(self, root: Optional[TreeNode]) -> bool:

        self.prev = -math.inf
        self.valid = True
        def inorder(root):

            if not root:
                return
            
            inorder(root.left)
            if self.prev >= root.val:
                self.valid = False
                return
            self.prev = root.val
            inorder(root.right)
        
        inorder(root)
        return self.valid

```

### [Number of Islands](https://leetcode.com/problems/number-of-islands/)<a name="number-of-islands"></a> 
```python
def numIslands(self, grid: List[List[str]]) -> int:

        dirs = [0,1,0,-1,0]
        def dfs(x,y):

            for i in range(4):
                nx,ny = x + dirs[i],y+dirs[i+1]

                if nx >=0 and ny >= 0 and nx < len(grid) and ny < len(grid[0]) and grid[nx][ny] == "1":
                    grid[nx][ny] = "2"
                    dfs(nx,ny)
        
        num_islands = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    num_islands+=1
                    grid[i][j] = "2"
                    dfs(i,j)
        return num_islands

```

### [Rotting Oranges](https://leetcode.com/problems/rotting-oranges/)<a name="rotting-oranges"></a>  

```python
def orangesRotting(self, grid: List[List[int]]) -> int:

        m = 0
        q = []
        dirs = [0,1,0,-1,0]
        ones = set()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 2:
                    q.append((i,j))
                if grid[i][j] == 1:
                    ones.add((i,j))
        

        while q:
            newq = []

            for x,y in q:
                for i in range(4):
                    nx,ny = x+dirs[i],y+dirs[i+1]
                    if nx >= 0 and ny >= 0 and nx < len(grid) and ny < len(grid[0]) and grid[nx][ny] == 1:
                        grid[nx][ny] = 2
                        newq.append((nx,ny))
                        ones.remove((nx,ny))

            q = newq
            if q:
                m += 1
        
        if len(ones) > 0:
            return -1
        
        return m
```

### [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)<a name="product-of-array-except-self"></a>

```python
def productExceptSelf(self, nums: List[int]) -> List[int]:

        prod_f = [1]
        prod_b = 1
        for i in range(len(nums)-1):
            prod_f.append(prod_f[-1]*nums[i])

        # ret = [1]*len(nums)
        
        for i in reversed(range(len(nums))):
            prod_f[i] = prod_f[i]*prod_b
            prod_b = prod_b*nums[i]


        #now for example the answer for 0 is prod_b[1] * prod_f[0]
        #the answer for 1 is prod_b[2]*prod_f[1] and so on 

        # ret = []
        # for i in range(len(nums)):
        #     ret.append(prod_f[i] *prod_b[i+1])
        return prod_f

```

### [Min Stack](https://leetcode.com/problems/min-stack/)<a name="min-stack"></a> 

```python
class MinStack:

    def __init__(self):

        self.stack = []
        self.min = [math.inf]
        

    def push(self, val: int) -> None:
        self.stack.append(val)
        self.min.append(min(self.min[-1], val))
        

    def pop(self) -> None:
        self.stack.pop()
        self.min.pop()

    def top(self) -> int:
        return self.stack[-1]
        

    def getMin(self) -> int:
        return self.min[-1]
        
```

### [Search in Rotated Array](https://leetcode.com/problems/search-in-rotated-array/)<a name="search-in-rotated-array"></a>  

```python
def search(self, nums: List[int], target: int) -> int:
        #I think there's a way to do this without finding the pivot
        #I do think finding the pivot is good practice though...

        #find pivot
        l = 0
        r = len(nums)-1
        while l < r:
            m = (l+r)//2
            if nums[m] > nums[r]:
                l = m+1
            elif nums[l] > nums[m]:
                r = m
            else:
                r = l
        
        pivot = l
        # print(pivot)

        l = 0
        r = len(nums)
        while l < r:
            m = (l+r)//2
            if nums[int((m+pivot)%len(nums))] == target:
                return int((m+pivot)%len(nums))
            elif nums[int((m+pivot)%len(nums))] > target:
                r = m -1
            else:
                l = m+1
        
        if nums[int((l+pivot)%len(nums))]== target:
            return int((l+pivot)%len(nums))
        
        return -1
```


### [Permutations](https://leetcode.com/problems/permutations/)<a name="permutations"></a>  
```python
def permute(self, nums: List[int]) -> List[List[int]]:
        
        self.perms = []
        self.set = set()
        def permsearch(l):
            if len(l) == len(nums):
                self.perms.append(copy.deepcopy(l))
                return
            
            for i in range(len(nums)):
                if nums[i] not in l:
                    l.append(nums[i])
                    self.set.add(nums[i])
                    permsearch(l)
                    l.pop()
                    self.set.remove(nums[i])
        
        permsearch([])
        return self.perms

```

### [Combination Sum](https://leetcode.com/problems/combination-sum/)<a name="combination-sum"></a> 
```python
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

        candidates.sort(reverse = True)
        self.ret = set()

        def search(l,s, j):

            if s == target:
                self.ret.add(tuple(l))
                return
            
            for i in range(j, len(candidates)):
                if candidates[i] + s <= target:
                    search(l+[candidates[i]], s+candidates[i], i)
        
        search([],0,0)
        return self.ret     
```

### [Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)<a name="lowest-common-ancestor-of-a-binary-tree"></a> 
```python
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        #present on both left and right
        if root == p or root==q:
            return root

        def search(root):
            if not root:
                return None
            elif root == q or root == p:
                return root
            
            l = search(root.left)
            r = search(root.right)

            if l and r:
                return root
            elif l:
                return l
            elif r:
                return r
        return search(root)
```


### [Merge Intervals](https://leetcode.com/problems/merge-intervals/)<a name="merge-intervals"></a> 
```python
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        

        intervals.sort()
        stack = [intervals[0]]

        for i in range(1,len(intervals)):
            
            b,e = intervals[i]
            
            if b <= stack[-1][1]:
                stack[-1][1] = max(e,stack[-1][1])
            else:
                stack.append(intervals[i])
        
        return stack
```
### [Time Based Key-Value Store](https://leetcode.com/problems/time-based-key-value-store/)<a name="time-based-key-value-store"></a> 
```python
class TimeMap:

    def __init__(self):
        self.times = defaultdict(list)
        self.dict = defaultdict(list)
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.times[key].append(timestamp)
        self.dict[key].append(value)
        

    def get(self, key: str, timestamp: int) -> str:

        if key not in self.times:
            return ""
        
        ind = bisect.bisect(self.times[key], timestamp)
        #bisect_right outputs the index of the first value greater than timestamp.

        if ind == 0:
            return ""
        else:
            return self.dict[key][ind-1]

```



### [Accounts Merge](https://leetcode.com/problems/accounts-merge/)<a name="accounts-merge"></a> 

```python
def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:

        #build graph

        emailToInd = defaultdict(list)
        for i in range(len(accounts)):
            for j in range(1,len(accounts[i])):
                emailToInd[accounts[i][j]].append(i)
        
        graph = defaultdict(list)
        for email in emailToInd:
            if len(emailToInd)>1:
                ind = emailToInd[email][0]
                for i in range(1,len(emailToInd[email])):
                    j = emailToInd[email][i]
                    graph[ind].append(j)
                    graph[j].append(ind)
        
        #find connected components
        self.visited = set()
        self.connected_components = []

        def bfs(ind):
            ret = [ind]
            q = collections.deque([ind])
            self.visited.add(ind)
            while q:
                n = q.popleft()
                for nbr in graph[n]:
                    if nbr not in self.visited:
                        q.append(nbr)
                        ret.append(nbr)
                        self.visited.add(nbr)
            
            return ret
        
        for i in range(len(accounts)):
            if i not in self.visited:
                self.connected_components.append(bfs(i))
        
        ret = []
        for l in self.connected_components:
            if len(l) == 1:
                ret.append([accounts[l[0]][0]] + sorted(list(set(accounts[l[0]][1:]))))
            else:
                tmp = set()
                for i in range(len(l)):
                    for j in range(1,len(accounts[l[i]])):
                        tmp.add(accounts[l[i]][j])
                
                ret.append([accounts[l[0]][0]] + sorted(list(tmp)))

        return ret
                
```

### [Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)<a name="spiral-matrix"></a> 
```python
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        
        lb = 0 
        rb = len(matrix[0])-1
        ub = 0
        bb = len(matrix)-1

        ret = []

        while lb <= rb and ub <= bb:

            for i in range(lb,rb + 1):
                ret.append(matrix[ub][i])
            ub += 1


            
            if ub <= bb:
                for i in range(ub,bb + 1):
                    
                    ret.append(matrix[i][rb])
                rb -= 1
            else:
                break
            
            # print(lb,rb)
            if lb <= rb:
                for i in reversed(range(lb,rb + 1)):
                    ret.append(matrix[bb][i])
                bb -= 1
            else:
                break
            if ub <= bb:
                for i in reversed(range(ub,bb + 1)):
                    ret.append(matrix[i][lb])
                lb += 1
            else:
                break
        
        return ret
```

### [String To Integer Atoi](https://leetcode.com/problems/string-to-integer-atoi/)<a name="string-to-integer-atoi"></a> 

```python
def myAtoi(self, s: str) -> int:

        #well let's give it a shot
        if s == "":
            return 0

        #first clear out whitespace:
        ind = 0
        while ind < len(s) and s[ind] == " ":
            ind += 1
        if ind == len(s):
            return 0
        neg = False
        if s[ind] == '-':
            neg = True
            ind += 1
        elif s[ind] == "+":
            ind += 1
        
        #start of number
        nums = set(list(("1234567890")))
        
        if ind == len(s) or s[ind] not in nums:
            return 0
        
        ret = 0
        next_ind = ind
        while next_ind < len(s):
            if s[next_ind] not in nums:
                break
            next_ind += 1
        
        ret = int(s[ind:next_ind])
        if neg:
            ret = -ret
        if neg:
            ret = max(ret, -2**31)
        else:
            ret = min(2**31 - 1, ret)
        
        return ret

```


### [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)<a name="partition-equal-subset-sum"></a> 

```python

def canPartition(self, nums: List[int]) -> bool:

        #ah, finally one I don't remember. Good thing we're reviewing.
        #my best guess is since order doesn't matter, we can sort the nums
        #although, I don't know precisely how that would help
        #(aside from speeding up the backtracking potentially) and then
        #we have to backtrack?
        #oh wait hang on, We can calculate the total sum, then find something that adds up
        #to sum(nums)//2

        t = sum(nums)
        if t % 2 == 1:
            return False
        target = t//2

        nums.sort(reverse = True) #take big steps initially?

        #this is basically the coin problem

        dp = [True] + [False]*target

        for n in nums:
            for i in reversed(range(n, len(dp))):
                dp[i] = dp[i] or dp[i-n]
        
        return dp[-1]
```


### [Word Break](https://leetcode.com/problems/word-break/)<a name="word-break"></a> 
```python
def wordBreak(self, s: str, wordDict: List[str]) -> bool:

        wordDict = set(wordDict)
        dp = [False]*(len(s)+1)
        dp[0] = True
        for i in range(1,len(s)+1):
            
            for j in range(i):
                dp[i] = dp[i] or (dp[j] and s[j:i] in wordDict)
                if dp[i]:
                    break
        
        return dp[-1]
```

### [Sort Colors](https://leetcode.com/problems/sort-colors/)<a name="sort-colors"></a> 
Honestly the single pass solution is actually slower, asymptotically and empirically.
```python
def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        # c = collections.Counter(nums)
        # ind = 0
        # for i in range(c[0]):
        #     nums[ind] = 0
        #     ind += 1
        # for i in range(c[1]):
        #     nums[ind] = 1
        #     ind += 1
        # for i in range(c[2]):
        #     nums[ind] = 2
        #     ind += 1

        #onepass with constant extra space?

        counts = [-1,-1,-1]

        for i in range(len(nums)):
            if nums[i] == 0:
                
                counts[0] += 1
                counts[1] += 1
                counts[2] += 1
                nums[counts[2]] = 2
                nums[counts[1]] = 1
                nums[counts[0]] = 0
                
            elif nums[i] == 1:
                counts[1] += 1
                counts[2] += 1
                nums[counts[2]] = 2
                nums[counts[1]] = 1
                
            else:
                counts[2] += 1
                nums[counts[2]] = 2
        return nums
```

### [Subsets](https://leetcode.com/problems/subsets/)<a name="subsets"></a> 
```python
def subsets(self, nums: List[int]) -> List[List[int]]:

        # nums.sort()

        #must use backtracking, try to be efficient
        self.ret = [[]]
        def search(l,ind):
            if ind == len(nums):
                return
        
            for i in range(ind, len(nums)):

                l.append(nums[i])
                self.ret.append(l.copy())
                search(l,i+1)
                l.pop()
        
        search([],0)
        return self.ret
```


### [Container With Most Water](https://leetcode.com/problems/container-with-most-water/)<a name="container-with-most-water"></a> 
```python
def maxArea(self, height: List[int]) -> int:

        water = 0
        l = 0
        r = len(height)-1
        while l < r:
            water = max(water, min(height[l],height[r]) * (r-l))
            if height[l] < height[r]: #anything including l that is smaller must have strictly smaller water
                l += 1
            else:
                r -= 1

        return water

```

### [Unique Paths](https://leetcode.com/problems/unique-paths/)<a name="unique-paths"></a> 
```python
def uniquePaths(self, m: int, n: int) -> int:
        

        #less space efficient (O(nm))
        # dp = [[0]*n for _ in range(m)]
        
        # for i in range(n):
        #     dp[0][i] = 1
        # for i in range(1,m):
        #     dp[i][0] = 1
        
        # for i in range(1,m):
        #     for j in range(1,n):
        #         dp[i][j] = dp[i-1][j] + dp[i][j-1]

        if m < n:
            return self.uniquePaths(n,m)
        dp = [1]*n
        for i in range(1,m):
            for j in range(1,n):
                dp[j] = dp[j-1] + dp[j]
            
        
        return dp[-1]
```


### [Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/)<a name="binary-tree-right-side-view"></a> 
```python
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        
        if not root:
            return []
        q = [root]
        ret = [root.val]
        while q:
            newq = []
            for n in q:
                if n.left:
                    newq.append(n.left)
                if n.right:
                    newq.append(n.right)
            
            if newq:
                ret.append(newq[-1].val)
            q = newq
        return ret
```

### [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)<a name="longest-palindromic-substring"></a> 
```python
def longestPalindrome(self, s: str) -> str:
        
        #could this be a stack problem maybe?
        #no
        #here's a thought.
        #go through and keep track of len one and len two substring using pairs
        #of indices, then just build on each one that you can until you can't anymore
        #not sure this would be faster, but it's worth a shot. it's definitely faster actually
        #but worse for memory

        palindromes = set([(i,i) for i in range(len(s))])

        mxstr = s[0]

        for i in range(len(s)-1):
            if s[i] == s[i+1]:
                palindromes.add((i,i+1))
                mxstr = s[i:i+2]
        
        while palindromes:
            # print(palindromes)
            newset = set()
            for i,j in palindromes:
                
                # palindromes.remove((i,j))
                if i > 0 and j < len(s)-1 and s[i-1] == s[j+1]:
                    newset.add((i-1,j+1))
                    if (j+2-(i-1)) > len(mxstr):
                        mxstr = s[i-1:j+2]
            palindromes = newset
        
        return mxstr
```

### [Construct Binary Tree From Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)<a name="construct-binary-tree-from-preorder-and-inorder-traversal"></a> 
```python
def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:

        if not preorder:
            return None

        root = TreeNode(preorder[0])
        ind = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:ind+1],inorder[:ind])
        root.right = self.buildTree(preorder[ind+1:],inorder[ind+1:])

        return root

```


### [Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)<a name="letter-combinations-of-a-phone-number"></a> 
```python
def letterCombinations(self, digits: str) -> List[str]:

        if not digits:
            return []

        digdict = {2:"abc", 3: "def", 4:"ghi",5:"jkl",6:"mno",7:"pqrs",8:"tuv",9:"wxyz"}

        self.ret = []
        # @cache
        def search(s,ind):
            if ind == len(digits):
                self.ret.append(s)
                return
            
            for l in digdict[int(digits[ind])]:
                search(s+l,ind+1)
        
        search("",0)
        return self.ret
```

### [Word Search](https://leetcode.com/problems/word-search/)<a name="word-search"></a> 
```python
def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """

        s1 = set(word)
        s2 = set(board[0])
        for i in range(1,len(board)):
            s2 = s2.union(set(board[i]))
        
        for l in s1:
            if l not in s2:
                return False

        def dfs(i, j, l, visited):

            if l == len(word):
                return True

            elif (i,j) in visited:
                return False

            elif board[i][j] != word[l]:
                return False

            elif board[i][j] == word[l] and l == len(word) - 1:
                return True

            dirs = [0,1,0,-1,0]
            bools = []
            visited.append((i,j))
            for k in range(4):
                if (0 <= i + dirs[k] < len(board)) and (0 <= j + dirs[k + 1] < len(board[0])):
                    bools.append(dfs(i + dirs[k], j + dirs[k+1], l + 1, visited))
            
            if any(bools):
                return True
            else:
                visited.pop()
            
            

        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i,j, 0, []):
                    return True

        return False

```

### [Find Anagrams](https://leetcode.com/problems/find-anagrams/)<a name="find-anagrams"></a> 
```python
def findAnagrams(self, s: str, p: str) -> List[int]:
        #great, more counting

        cp = collections.Counter(p)
        
        c = collections.Counter(s[:len(p)])
        ret = []
        for i in range(len(p),len(s)):
            if c == cp:
                ret.append(i-len(p))
            c[s[i-len(p)]] -= 1
            c[s[i]] += 1
        if c == cp:
            ret.append(len(s)-len(p))
        return ret
```

### [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)<a name="daily-temperatures"></a>
```python
def dailyTemperatures(self, temperatures: List[int]) -> List[int]:


        #this is a stack question I believe. 
        # you want to annihilate elements on end of stack when you're temp is greater
        #and keep track of inds at the same time

        stack = []

        answer = [0]*len(temperatures)

        for i,t in enumerate(temperatures):

            if not stack or t <= stack[-1][0]:
                stack.append((t,i))
            else:
                while stack and t > stack[-1][0]:
                    answer[stack[-1][1]] = i - stack[-1][1]
                    stack.pop()
                
                stack.append((t,i))
                
        return answer
```

### [Kth Smallest Element in Binary Search Tree](https://leetcode.com/problems/kth-smallest-element-in-a-bs/)<a name="kth-smallest-element-in-a-bs"></a>

```python
def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:

        self.cnt = 0
        self.val = 0
        def inorder(root):
            if not root:
                return
            inorder(root.left)
            
            self.cnt +=1
            if self.cnt == k:
                self.val = root.val
                return
            
            inorder(root.right)
        
        inorder(root)
        return self.val

```

### [House Robber](https://leetcode.com/problems/house-robber/)<a name="house-robber"></a>

```python
def rob(self, nums: List[int]) -> int:

        if len(nums) == 1:
            return nums[0]
        elif len(nums) == 2:
            return max(nums)
        elif len(nums) == 3:
            return max(nums[1], nums[0]+ nums[2])
        prevprev = nums[0]
        prev = nums[1]
        curr = prevprev + nums[2]


        for i in range(3,len(nums)):
            nxt = max(prevprev + nums[i], prev +nums[i])
            prevprev = prev
            prev = curr
            curr = nxt
        
        return max(curr,prev)
```

### [Task Scheduler](https://leetcode.com/problems/task-scheduler/)<a name="task-scheduler"></a>
```python
def leastInterval(self, tasks: List[str], n: int) -> int:



        c = collections.Counter(tasks)

        max_val = max(v for v in c.values())

        tasklen = (max_val -1)*(n+1)
        nmax = sum(v == max_val for v in c.values())

        #the intuition here is that you need max_val-1 full cycles of length n+1 to clear out 
        #the tasks, nmax is the runover after those cycles
        #still hazy on the intuition as to why the below works honestly. I prefer my other solution 
        #in terms of clarity

        return max(len(tasks), tasklen + nmax)


        taskorder = sorted(c.keys(),key = lambda x:c[x], reverse = True)



        most_recent = {k:-1 for k in c}

        tasks = []
        while c:
            found = False
            for k in taskorder:
                if k in c and (most_recent[k] == -1 or len(tasks)-most_recent[k] > n):
                    tasks.append(k)
                    most_recent[k] = len(tasks)-1
                    found = True
                    c[k] -= 1
                    if c[k] == 0:
                        del c[k], most_recent[k]
                    break
            if not found:
                tasks.append(0)
            taskorder = sorted(c.keys(),key = lambda x:c[x], reverse = True)
        return len(tasks)
```


### [Minimum Height Trees](https://leetcode.com/problems/minimum-height-trees/)<a name="minimum-height-trees"></a>
```python
def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:

        #the intuition here: do not pick a leaf. Pick something as far away from
        #all leaves as possible. sort of a bfs

        if n == 1:
            return [0]

        graph = defaultdict(set)
        for x,y in edges:
            graph[x].add(y)
            graph[y].add(x)

        leaves = []

        for n in graph:
            if len(graph[n]) == 1:
                leaves.append(n)

        ret = leaves
        while graph:
            newleaves = []
            for n in leaves:
                if len(graph[n]) > 0:
                    checkleaf = next(iter(graph[n]))
                    if checkleaf in graph:
                        graph[checkleaf].remove(n)
                        
                        if len(graph[checkleaf])==1:
                            newleaves.append(checkleaf)
                del graph[n]
            if newleaves:
                ret = newleaves
            leaves = newleaves
        return ret
```

### [Gas Station](https://leetcode.com/problems/gas-station/)<a name="gas-station"></a>
This is definitely worth doing again.
```python
def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:

        if sum(cost) > sum(gas):
            return -1
        
        s = 0
        ind = 0
        for i in range(len(gas)):
            s += gas[i]-cost[i]
            if s < 0:
                s = 0
                ind = i + 1
        return ind

```


### [Valid Sudoku](https://leetcode.com/problems/valid-sudoku/)<a name="valid-sudoku"></a>
```python
def isValidSudoku(self, board: List[List[str]]) -> bool:
        
        for i in range(9):
            c = collections.Counter(board[i])
            for k in c:
                if k != ".":
                    if c[k] > 1:
                        
                        return False
        
        for i in range(9):
            c = collections.Counter()
            for j in range(9):
                if board[j][i] != ".":
                    k = board[j][i]
                    c[k] += 1
                    if c[k] > 1:
                        return False

        for i in range(3):
            for j in range(3):
                c = collections.Counter()
                for ii in range(3):
                    for jj in range(3):
                        indi = (3*i) + ii
                        indj = (3*j) + jj
                        k = board[indi][indj]
                        if k != ".":
                            c[k] += 1
                            if c[k] > 1:
                                # print(i,j,ii,jj, indi, indj, c)
                                return False

        return True
```

### [Next Permutation](https://leetcode.com/problems/next-permutation/)<a name="next-permutation"></a>
```python
def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """


        ind = len(nums)-2
        while ind >=0:
            if nums[ind] < nums[ind+1]:
                break
            ind -=1
        
        if ind != -1:
            for i in reversed(range(ind+1,len(nums))):
                if nums[i] > nums[ind]:
                    nums[i],nums[ind] = nums[ind],nums[i]
                    break
                    # print(i,ind)
     
        
        l = ind + 1
        r = len(nums)-1
        while l < r:
            nums[l],nums[r] = nums[r], nums[l]
            l +=1
            r -=1 

```

### [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)<a name="maximum-product-subarray"></a>

```python
def maxProduct(self, nums: List[int]) -> int:

        
        mn = 1
        mx = 1
        ret = -math.inf

        #we want to track the maximum value including i and the minimum value including i
        #should be able to do it without an array...
        for i in range(len(nums)):
            newmx = max(nums[i], mx*nums[i], mn*nums[i])
            newmn = min(nums[i],mx*nums[i],mn*nums[i])
            mx,mn = newmx,newmn
            ret = max(ret,mx)
        return ret

```

### [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)<a name="remove-nth-node-from-end-of-list"></a>

```python
def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:


        fast = head
        slow = head
        cnt = 0
        while cnt < n:
            fast = fast.next
            cnt +=1
        
        if not fast:
            return head.next
        
        while fast and fast.next:
            fast = fast.next
            slow = slow.next
        
        #remove slow.next
        slow.next = slow.next.next
        return head
```

### [Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)<a name="design-add-and-search-words-data-structure"></a>  

```python
class TrieNode():
    def __init__(self):
        self.children = {}
        self.word = False

class WordDictionary:

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for l in word:
            if l not in node.children:
                node.children[l] = TrieNode()
            node = node.children[l]
        node.word = True
        

    def search(self, word: str) -> bool:

        return self.dfs(self.root, word, 0)
    
    def dfs(self, node, word, ind):

        if ind == len(word):
            return node.word

        l = word[ind]
        if l != ".":
            if l in node.children:
                if self.dfs(node.children[l], word, ind+1):
                    return True
            
        else:
            for nl in node.children:
                if self.dfs(node.children[nl], word, ind+1):
                    return True
        
        return False
```


### [Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)<a name="pacific-atlantic-water-flow"></a>

```python
def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:

        #do a bfs from points on the boundary 

        if len(heights) == 1 or len(heights[0]) == 1:
            ret = set()
            for i in range(len(heights)):
                for j in range(len(heights[0])):
                    ret.add((i,j))
            return ret

        self.ret = set([(0,len(heights[0])-1), (len(heights)-1, 0)])

        flow = [[0]*len(heights[0]) for _ in range(len(heights))]
        for i in range(len(heights[0])):
            flow[0][i] = 'P'
            flow[len(heights)-1][i] = 'A'
        for i in range(len(heights)):
            if flow[i][0] == 'A':
                flow[i][0] = 'B'
                self.ret.add((i,0))
            else:
                flow[i][0] = 'P'
            if flow[i][len(heights[0])-1] == 'P':
                flow[i][len(heights[0])-1] = 'B'
                self.ret.add((i,len(heights[0])-1))
            else:
                flow[i][len(heights[0])-1] = 'A'
        
        # flow[0][len(heights[0])-1] = 'B'
        # flow[len(heights)-1][0] = 'B'

        def bfs(x, y):
            visited = set([(x,y)])
            dirs = [0,1,0,-1,0]
            q = collections.deque([(x,y)])
            while q:
                
                i,j = q.popleft()
                for k in range(4):
                    ni,nj = i +dirs[k], j+dirs[k+1]
                    if ni >= 0 and nj >= 0 and ni < len(heights) and nj < len(heights[0]) and heights[ni][nj] >= heights[i][j] and (ni,nj) not in visited:
                        visited.add((ni,nj))
                        q.append((ni,nj))
                        if flow[ni][nj] == 0:
                            flow[ni][nj] = flow[i][j]
                            if flow[ni][nj] == 'B':
                                self.ret.add((ni,nj))
                        elif flow[ni][nj] != flow[i][j]:
                            flow[ni][nj] = 'B'
                            self.ret.add((ni,nj))
        
        
        for i in range(len(heights[0])):
            bfs(0,i)
            bfs(len(heights)-1,i)
        for i in range(len(heights)):
            bfs(i,0)
            bfs(i, len(heights[0])-1)
        

        return self.ret
            
```

### [Group Anagrams](https://leetcode.com/problems/group-anagrams/)<a name="group-anagrams"></a>
```python
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:


        
        cosets = set(["".join(sorted(s)) for s in strs])

        ret = [[] for _ in range(len(cosets))]
        cnt = 0
        an_map = {}
        for c in cosets:
            an_map[c] = cnt
            cnt += 1
        for s in strs:
            ret[an_map["".join(sorted(s))]].append(s)
        return ret

```

### [Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)<a name="swap-nodes-in-pairs"></a>
```python
def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:

        if not head or not head.next:
            return head
        
        dummy = ListNode(-1, head)

        prev = dummy
        f1 = head
        f2 = head.next
        nxt = head.next.next

        while prev != None and f1 != None and f2 != None:
            f2.next = f1
            f1.next = nxt
            prev.next = f2
            prev = f1
            f1 = nxt
            if f1 == None:
                break
            f2 = f1.next
            if f2 == None:
                break
            nxt = f2.next

        
        return dummy.next
```










