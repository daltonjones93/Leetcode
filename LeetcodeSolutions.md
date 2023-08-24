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





