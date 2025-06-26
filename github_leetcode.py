from typing import Optional
import pandas as pd
class Solution:
    # 182. Duplicate Emails (Easy)
    def duplicate_emails(person: pd.DataFrame) -> pd.DataFrame:
        return person.groupby("Email").filter(lambda x: len(x) > 1)[["Email"]].drop_duplicates()

    # 183. Customers Who Never Order (Easy)
    def customers_who_never_order(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
        return customers[~customers["Id"].isin(orders["CustomerId"])][["Name"]].rename(columns={"Name": "Customers"})

    # 577. Employee Bonus (Easy)
    def employee_bonus(employee: pd.DataFrame, bonus: pd.DataFrame) -> pd.DataFrame:
        df = employee.merge(bonus, left_on='id', right_on='empId', how='left')
        df = df[['name', 'bonus']].fillna(0)
        return df[df['bonus'] < 1000]

    #1.Two Sum
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict_ = {}
        for i, num in enumerate(nums):
            sub = target - num
            if sub in dict_:
                return [dict_[sub], i]
            else:
                dict_[num] = i
        return []
    #367. Valid Perfect Square
    def validPerfectSquare(self, num:int) ->bool:
        num = num**0.5
        perfect_num = num//1
        return num == perfect_num
    #415. Add Strings
    def addStrings(self, num1: str, num2: str) -> str:
        i, j = len(num1) -1, len(num2)-1
        carry = 0
        result = []
        while(i >= 0 or j >= 0 or carry):
            digit1 = int(num1[i]) if i >= 0 else 0
            digit2 = int(num2[j]) if j >= 0 else 0
            total = digit1 + digit2 + carry
            carry = total // 10
            result.append(str(total % 10))
            i -=1
            j -=1
        return ''.join(result[::-1])
    #509. Fibonacci Number
    def fib(self, n: int) -> int:
        l = [0, 1]
        for i in range(1,n):
            l.append(l[i] + l[i-1])
        return l[n]
    #704. Binary Search
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return -1 #If not in the list, return -1
    #824. Goat Latin
    def toGoatLatin(self, sentence: str) -> str:
        vowels = ['a', 'e', 'i', 'o', 'u']
        final_sentence = ""
        words_list = sentence.split(" ")
        for word_index, word in enumerate(words_list):
            if word[0].lower() in vowels:
                new_word = word + "ma"
            else:
                first_letter = word[0]
                new_word = word[1:]
                new_word = new_word + first_letter + "ma"
            for i in range(0, word_index+1):
                new_word += "a"
            final_sentence += new_word + " "
        return final_sentence.strip()
    #268. Missing Number
    def missingNumber(self, nums:int) -> int:
        for i in range(len(nums)+1):
            if i not in nums:
                return i
    #476. Number Complement
    def findComplement(self, num: int) -> int:  
        list_num = list(format(num, 'b'))
        for i in range(len(list_num)):
            if int(list_num[i]) == 1:
                list_num[i] = '0'
            else:
                list_num[i] = '1'
        return int(''.join(list_num), 2)
    # 2. Add Two Numbers (Medium)
    def addTwoNumbers(l1, l2):
        dummy = ListNode(0)
        curr, carry = dummy, 0
        while l1 or l2 or carry:
            carry += (l1.val if l1 else 0) + (l2.val if l2 else 0)
            curr.next = ListNode(carry % 10)
            carry //= 10
            curr, l1, l2 = curr.next, l1.next if l1 else None, l2.next if l2 else None
        return dummy.next   

    # 4. Median of Two Sorted Arrays (Hard)
    def findMedianSortedArrays(nums1, nums2):
        nums = sorted(nums1 + nums2)
        mid = len(nums) // 2
        return (nums[mid] + nums[mid - 1]) / 2 if len(nums) % 2 == 0 else nums[mid]

    # 9. Palindrome Number (Easy)
    def isPalindrome(x):
        return str(x) == str(x)[::-1]

    # 13. Roman to Integer (Easy)
    def romanToInt(s):
        roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        for i in range(len(s)):
            total += -roman[s[i]] if i < len(s)-1 and roman[s[i]] < roman[s[i+1]] else roman[s[i]]
        return total

    # 14. Longest Common Prefix (Easy)
    def longestCommonPrefix(strs):
        prefix = strs[0]
        for s in strs[1:]:
            while not s.startswith(prefix):
                prefix = prefix[:-1]
        return prefix

    # 15. 3Sum (Medium)
    def threeSum(nums):
        nums.sort()
        res, n = [], len(nums)
        for i in range(n-2):
            if i > 0 and nums[i] == nums[i-1]: continue
            l, r = i+1, n-1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s < 0: l += 1
                elif s > 0: r -= 1
                else:
                    res.append([nums[i], nums[l], nums[r]])
                    l += 1
                    while l < r and nums[l] == nums[l-1]: l += 1
        return res

    # 21. Merge Two Sorted Lists (Easy)
    def mergeTwoLists(l1, l2):
        if not l1 or not l2: return l1 or l2
        if l1.val > l2.val: l1, l2 = l2, l1
        l1.next = mergeTwoLists(l1.next, l2)
        return l1

    # 26. Remove Duplicates from Sorted Array (Easy)
    def removeDuplicates(nums):
        i = 0
        for n in nums:
            if i < 1 or nums[i-1] != n:
                nums[i] = n
                i += 1
        return i

    # 27. Remove Element (Easy)
    def removeElement(nums, val):
        i = 0
        for n in nums:
            if n != val:
                nums[i] = n
                i += 1
        return i

    # 29. Divide Two Integers (Medium)
    def divide(dividend, divisor):
        return min(max(dividend // divisor, -2147483648), 2147483647)

    # 35. Search Insert Position (Easy)
    def searchInsert(nums, target):
        l, r = 0, len(nums)
        while l < r:
            mid = (l + r) // 2
            if nums[mid] < target:
                l = mid + 1
            else:
                r = mid
        return l

    # 50. Pow(x, n) (Medium)
    def myPow(x, n):
        return x ** n

    # 58. Length of Last Word (Easy)
    def lengthOfLastWord(s):
        return len(s.strip().split()[-1])

    # 66. Plus One (Easy)
    def plusOne(digits):
        return list(map(int, str(int("".join(map(str, digits))) + 1)))

    # 67. Add Binary (Easy)
    def addBinary(a, b):
        return bin(int(a, 2) + int(b, 2))[2:]

    # 69. Sqrt(x) (Easy)
    def mySqrt(x):
        return int(x ** 0.5)

    # 70. Climbing Stairs (Easy)
    def climbStairs(n):
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    # 73. Set Matrix Zeroes (Medium)
    def setZeroes(matrix):
        zero_rows, zero_cols = set(), set()
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    zero_rows.add(i)
                    zero_cols.add(j)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i in zero_rows or j in zero_cols:
                    matrix[i][j] = 0

    # 74. Search a 2D Matrix (Medium)
    def searchMatrix(matrix, target):
        return any(target in row for row in matrix)

    # 83. Remove Duplicates from Sorted List (Easy)
    def deleteDuplicates(head):
        curr = head
        while curr and curr.next:
            if curr.val == curr.next.val:
                curr.next = curr.next.next
            else:
                curr = curr.next
        return head

    # 88. Merge Sorted Array (Easy)
    def merge(nums1, m, nums2, n):
        nums1[m:] = nums2
        nums1.sort()

    # 100. Same Tree (Easy)
    def isSameTree(p, q):
        if not p or not q: return p is q
        return p.val == q.val and isSameTree(p.left, q.left) and isSameTree(p.right, q.right)

    # 110. Balanced Binary Tree (Easy)
    def isBalanced(root):
        def height(node):
            if not node: return 0
            left, right = height(node.left), height(node.right)
            return max(left, right) + 1 if abs(left - right) <= 1 else -1
        return height(root) != -1

    # 136. Single Number (Easy)
    def singleNumber(nums):
        res = 0
        for n in nums:
            res ^= n
        return res

    # 202. Happy Number (Easy)
    def isHappy(n):
        seen = set()
        while n != 1 and n not in seen:
            seen.add(n)
            n = sum(int(digit) ** 2 for digit in str(n))
        return n == 1

    # 222. Count Complete Tree Nodes (Easy)
    def countNodes(root):
        return 1 + countNodes(root.left) + countNodes(root.right) if root else 0

    # 228. Summary Ranges (Easy)
    def summaryRanges(nums):
        res, i = [], 0
        while i < len(nums):
            start = nums[i]
            while i + 1 < len(nums) and nums[i] + 1 == nums[i + 1]:
                i += 1
            res.append(str(start) if start == nums[i] else f"{start}->{nums[i]}")
            i += 1
        return res

    # 231. Power of Two (Easy)
    def isPowerOfTwo(n):
        return n > 0 and n & (n - 1) == 0

    # 258. Add Digits (Easy)
    def addDigits(num):
        return 1 + (num - 1) % 9 if num else 0

    # 263. Ugly Number (Easy)
    def isUgly(n):
        if n <= 0: return False
        for i in [2, 3, 5]:
            while n % i == 0:
                n //= i
        return n == 1

    # 283. Move Zeroes (Easy)
    def moveZeroes(nums):
        nums.sort(key=lambda x: x == 0)

    # 326. Power of Three (Easy)
    def isPowerOfThree(n):
        return n > 0 and 1162261467 % n == 0  # 3^19 is the max power of 3 within int range

    # 338. Counting Bits (Easy)
    def countBits(n):
        return [bin(i).count('1') for i in range(n + 1)]

    # 342. Power of Four (Easy)
    def isPowerOfFour(n):
        return n > 0 and n & (n - 1) == 0 and (n - 1) % 3 == 0

    # 344. Reverse String (Easy)
    def reverseString(s):
        s.reverse()

    # 383. Ransom Note (Easy)
    def canConstruct(ransomNote, magazine):
        return all(ransomNote.count(c) <= magazine.count(c) for c in set(ransomNote))

    # 389. Find the Difference (Easy)
    def findTheDifference(s, t):
        return chr(sum(map(ord, t)) - sum(map(ord, s)))

    # 414. Third Maximum Number (Easy)
    def thirdMax(nums):
        nums = sorted(set(nums), reverse=True)
        return nums[2] if len(nums) > 2 else nums[0]

    # 455. Assign Cookies (Easy)
    def findContentChildren(g, s):
        g.sort()
        s.sort()
        i, j = 0, 0
        while i < len(g) and j < len(s):
            if s[j] >= g[i]:
                i += 1
            j += 1
        return i

    # 507. Perfect Number (Easy)
    def checkPerfectNumber(num):
        return num > 1 and sum(i for i in range(1, int(num**0.5) + 1) if num % i == 0 and i != num) * 2 == num

    # 1051. Height Checker (Easy)
    def heightChecker(heights):
        return sum(a != b for a, b in zip(heights, sorted(heights)))

    # 1523. Count Odd Numbers in an Interval Range (Easy)
    def countOdds(low, high):
        return (high - low) // 2 + (low % 2 or high % 2)

    # 2410. Maximum Matching of Players With Trainers (Medium)
    def matchPlayersAndTrainers(players, trainers):
        players.sort()
        trainers.sort()
        i, j = 0, 0
        while i < len(players) and j < len(trainers):
            if players[i] <= trainers[j]:
                i += 1
            j += 1
        return i

    # 2914. Minimum Number of Changes to Make Binary String... (Medium)
    def minChanges(s):
        return sum(i % 2 == int(c) for i, c in enumerate(s))

    #860. LemonadeChange
    def lemonadeChange(self, bills: list[int]) -> bool:
        five, ten = 0, 0  # Counters for $5 and $10 bills
        
        for bill in bills:
            if bill == 5:  
                five += 1  # Accept $5
            elif bill == 10:
                if five == 0:  # Need one $5 to give change
                    return False
                five -= 1
                ten += 1  # Accept $10
            else:  # When bill == 20
                if ten > 0 and five > 0:  
                    ten -= 1  # Use one $10
                    five -= 1  # Use one $5
                elif five >= 3:  
                    five -= 3  # Use three $5 bills
                else:
                    return False  # Cannot give change
        return True
    #1154.Day of the year
    def dayOfYear(self, date: str) -> int:
            y, m, d = map(int, date.split("-"))
            days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if (y % 400) == 0 or ((y % 4 == 0) and (y % 100 != 0)): days[1] = 29
            return d + sum(days[:m-1])
    #1323.Maximum 69 number
    def maximum69Number (self, num: int) -> int:
            s = list(str(num))
            if '6' not in s: return num
            s[s.index('6')] = '9'
            return int(''.join(s))
    #1518.Water bottles
    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        n = numBottles
        m = numExchange
        count = 0
        while n >= m:
            n -= m
            n += 1
            count += m
        return count + n
    #7. Reversed Integer
    def reverse(self, x: int) -> int:
        sign = -1 if x < 0 else 1
        rev = int(str(abs(x))[::-1]) * sign
        return rev if -(2**31) <= rev <= (2**31 - 1) else 0
    #709. To lower case
    def toLowerCase(self, s: str) -> str:
            return s.lower()
    #1046. Last Stone Weight
    def lastStoneWeight(self, stones: list[int]) -> int:
            if len(stones) == 0:
                return 0
            if len(stones) == 1:
                return stones[0]
            stones.sort(reverse=True)
            if(stones[0] == stones[1]):
                return self.lastStoneWeight(stones[2:])
            else:
                stones.append(abs(stones[0]-stones[1]))
                return self.lastStoneWeight(stones[2:])
    #412. FizzBuzz
    def fizzBuzz(self, n: int) -> List[str]:
            result = []
            for i in range(1,n+1):
                if i % 5 == 0 and i % 3 == 0:
                    result.append("FizzBuzz")
                elif i % 3 == 0:
                    result.append("Fizz")
                elif i % 5 == 0:
                    result.append("Buzz")
                else:
                    result.append(str(i))
            return result
    #1507. Reformat Date
    def reformatDate(self, date: str) -> str:
            month_map = {
                "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", 
                "May": "05", "Jun": "06", "Jul": "07", "Aug": "08", 
                "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
            }
            date_str_list = date.split(" ")
            return(f"{date_str_list[2]}-{month_map[date_str_list[1]]}-{int(date_str_list[0][:-2]):02}")
    #1556. Thousand Separator
    def thousandSeparator(self, n: int) -> str:
            result = []
            str_n = str(n)
            if n >= 1000:
                for i in range(1, len(str_n)+1):
                    if i % 3 == 0 and i != len(str_n):
                        result.append(f".{str(str_n[::-1][i-1])}")                    
                    else:
                        result.append(str(str_n[::-1][i-1]))
                return(''.join(result[::-1]))
            else:
                return(str(n))
    #771. Jewels and Stones
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
            sum = 0
            for j in jewels:
                sum += stones.count(j)
            return sum
    #20. Valid Parentheses
    def isValid(self, s: str) -> bool:
            while len(s)>0:
                l = len(s)
                s = s.replace('()', '').replace('[]', '').replace('{}', '')
                if l==len(s): return False
            return True
    #506.Relative Ranks
    def findRelativeRanks(self, score: list[int]) -> list[str]:
        sorted_scores = sorted(score, reverse=True)
        rank_map = {}
        for i,s in enumerate(sorted_scores):
            if i == 0:
                rank_map[s] = "Gold Medal"
            elif i == 1:
                rank_map[s] = "Silver Medal"
            elif i == 2:
                rank_map[s] = "Bronze Medal"
            else:
                rank_map[s] = str(i+1)
        res = []
        for i in range(len(score)):
            res.append(rank_map[score[i]])
        return res
    #463. Island Perimeter
    def islandPerimeter(self, grid: list[list[int]]) -> int:
        perimeter = 0
        rows, cols = len(grid), len(grid[0])
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    perimeter += 4
                    if r > 0 and grid[r-1][c] == 1:
                        perimeter -= 2
                    if c > 0 and grid[r][c-1] == 1:
                        perimeter -= 2
        return perimeter
    #217. Contains Duplicate
    def containsDuplicate(self, nums: list[int]) -> bool:
            nums.sort()
            for i in range(1,len(nums)):
                if nums[i]==nums[i-1]:
                    return True
            return False
    #28. Find the Index of the First Occurrence in a String
    def strStr(self, haystack: str, needle: str) -> int:
            result = []
            i = 0
            while i < len(haystack):
                if haystack[i:i+len(needle)] == needle:
                    result.append(needle)
                    i += len(needle)
                else:
                    result.append(haystack[i])
                    i += 1
            a = result.index(needle) if needle in result else -1
            return a
    #118. Pascal's Triangle
    def generate(self, numRows: int) -> list[list[int]]:
        if numRows <= 0:
            return []
        result = [[1]]
        for i in range(1, numRows):
            rows = [1]
            for j in range(1,i):
                rows.append(result[i-1][j-1]+result[i-1][j])
            rows.append(1)
            result.append(rows)
        return result
    #119. Pascal's Triangle II
    def getRow(self, rowIndex: int) -> list[int]:
        rowNums = rowIndex+2
        if rowNums <= 0:
            return []
        result = [[1]]
        for i in range(1, rowNums):
            row = [1]
            for j in range(1, i):
                row.append(result[i-1][j-1] + result[i-1][j])
            row.append(1)
            result.append(row)
        return result[rowIndex]
    #43. Multiply String
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == '0' or num2 == '0':
            return '0'
        n1 = n2 = 0
        dic = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
        for num in num1:
            n1 *= 10
            n1 += dic[num]
        for num in num2:
            n2 *= 10
            n2 += dic[num]
        num_list = ['0','1','2','3','4','5','6','7','8','9']
        product = n1 * n2
        result = ''
        while product:
            val = product % 10
            result += num_list[val]
            product //= 10
        if result == '':
            return 0
        return result[::-1]
    #34. Find First and Last Position of Element in Sorted Array
    def searchRange(self, nums: list[int], target: int) -> list[int]:
        temp = []
        for index,i in enumerate(nums):
            if i == target:
                temp.append(index)
        if len(temp) == 1:
            temp.append(temp[0])
        if not temp:
            return [-1,-1]
        return temp[::len(temp)-1]
    #101. Symmetric Tree
    def isSymmetric(self, root: Optional[TreeNode]) -> bool: # type: ignore
        def isSym(a,b):
            if a is None and b is not None:
                return False
            if a is not None and b is None:
                return False
            if a is None and b is None:
                return True
            if a.val != b.val:
                return False
            return isSym(a.left, b.right) and isSym(a.right, b.left)
        return isSym(root.left, root.right)
    #94. Binary Tree Inorder Traversa
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]: # type: ignore
        ans=[]
        def inorder(root,ans):
            if not root:
                return None
            inorder(root.left,ans)
            ans.append(root.val)
            inorder(root.right,ans)
        inorder(root,ans)
        return ans
    #104. Maximum Depth of Binary Tree
    def maxDepth(self, root: Optional[TreeNode]) -> int: # type: ignore
        if not root:
            return 0
        l_depth = self.maxDepth(root.left)
        r_depth = self.maxDepth(root.right)
        return max(l_depth, r_depth) + 1
    #125. Valid Palindrome
    def isPalindrome(self, s: str) -> bool:
        s_clean = ''.join(c for c in s if c.isalnum()).lower()
        s_clean_rev = ''.join(c for c in s_clean[::-1])
        if s_clean == s_clean_rev:
            return True
        else:
            return False
    #121. Best Time to Buy and Sell Stock
    def maxProfit(self, prices: list[int]) -> int:
        if len(prices) == 0:
            return 0
        else:
            profit = 0
            minBuy = prices[0]
            for i in range(len(prices)):
                profit = max(prices[i] - minBuy, profit)
                minBuy = min(minBuy, prices[i])
            return profit
    #108. Convert Sorted Array to Binary Search Tree
    def sortedArrayToBST(self, nums: list[int]) -> Optional[TreeNode]: # type: ignore
        def build(l:int, r:int):
            if l > r:
                return None
            m = (l + r) // 2
            return TreeNode(nums[m], build(l, m-1), build(m+1, r)) # type: ignore
        return build(0, len(nums)-1)
    #111. Minimum Depth of Binary Tree
    def minDepth(self, root: Optional[TreeNode]) -> int: # type: ignore
        if not root:
            return 0
        l_depth = self.minDepth(root.left)
        r_depth = self.minDepth(root.right)
        if l_depth == 0 or r_depth == 0:
            return l_depth + r_depth + 1
        return 1 + min(l_depth, r_depth)
    #112. Path Sum
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool: # type: ignore
        def check(node, curSum):
            if not node:
                return False
            curSum += node.val
            if not node.left and not node.right:
                if curSum == targetSum:
                    return True
                else:
                    return False
            return check(node.left, curSum) or check(node.right, curSum)
        return check(root, 0)
    #113. Path Sum II
    def hasPathSumII(self, root: Optional[TreeNode], targetSum: int) -> list[list[int]]: # type: ignore
        result = []
        temp = []
        def check(node):
            if not node:
                return []
            temp.append(node.val)
            if not node.left and not node.right and sum(temp) == targetSum:
                result.append(temp.copy())
            else:
                check(node.left)
                check(node.right)
                temp.pop()
        check(root)
        return result
    #120. Triangle
    def minimumTotal(self, triangle: list[list[int]]) -> int:
        if not triangle:
            return 0
        for i in range(len(triangle) - 2, -1, -1):
            for j in range(len(triangle[i])):
                triangle[i][j] += min(triangle[i + 1][j], triangle[i + 1][j + 1])
        return triangle[0][0]
    #141. Linked List Cycle
    def hasCycle(self, head: Optional[ListNode]) -> bool: # type: ignore
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
    #144. Binary Tree Preorder Traversal
    def preorderTraversal(self, root: Optional[TreeNode]) -> list[int]: # type: ignore
        result = []    
        def check(root, result):
            if not root:
                return []
            result.append(root.val)
            check(root.left, result)
            check(root.right, result)
        check(root, result)
        return result
    #145. Binary Tree Postorder Traversal
    def postorderTraversal(self, root: Optional[TreeNode]) -> list[int]: # type: ignore
        result = []
        def check(root, result):
            if not root:
                return []
            check(root.left, result)
            check(root.right, result)
            result.append(root.val)
        check(root, result)
        return result
    #205. Isomorphic Strings
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        mapping_s_t, mapping_t_s = {}, {}
        for char_s, char_t in zip(s, t):
            if char_s not in mapping_s_t:
                mapping_s_t[char_s] = char_t
            if char_t not in mapping_t_s:
                mapping_t_s[char_t] = char_s
            if mapping_s_t[char_s] != char_t or mapping_t_s[char_t] != char_s:
                return False
        return True
    #219. Contains Duplicate II
    def containsNearbyDuplicate(self, nums: list[int], k: int) -> bool:
        index_map = {}
        for i, num in enumerate(nums):
            if num in index_map and i - index_map[num] <= k:
                return True
            index_map[num] = i
        return False
    #225. Implement Stack using Queues
    class MyStack:
        def __init__(self):
            self.queue = []

        def push(self, x: int) -> None:
            self.queue.append(x)

        def pop(self) -> int:
            if not self.queue:
                return None
            return self.queue.pop()

        def top(self) -> int:
            if not self.queue:
                return None
            return self.queue[-1]

        def empty(self) -> bool:
            return len(self.queue) == 0
    #3. Longest Substring Without Repeating Characters
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_map = {}
        left = max_length = 0
        for right, char in enumerate(s):
            if char in char_map and char_map[char] >= left:
                left = char_map[char] + 1
            char_map[char] = right
            max_length = max(max_length, right - left + 1)
        return max_length
    #8. String to Integer (atoi)
    def myAtoi(self, s: str) -> int:
        s = s.strip()
        if not s:
            return 0
        sign = 1
        if s[0] in ('-', '+'):
            if s[0] == '-':
                sign = -1
            s = s[1:]
        num = 0
        for char in s:
            if char.isdigit():
                num = num * 10 + int(char)
            else:
                break
        num *= sign
        return max(min(num, 2**31 - 1), -2**31)
    #242. Valid Anagram
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        count = {}
        for char in s:
            count[char] = count.get(char) + 1
        for char in t:
            if char not in count or count[char] == 0:
                return False
            count[char] -= 1
        return True
    #257. Binary Tree Paths
    def binaryTreePaths(self, root: Optional[TreeNode]) -> list[str]: # type: ignore
        def cont_path(root, path):
            if root:
                path += str(root.val)
                if not root.left and not root.right:
                    paths.append(path)
                else:
                    path += "->"
                    cont_path(root.left, path)
                    cont_path(root.right, path)
        paths = []
        cont_path(root, '')
        return paths
    #278.First Bad Version
    def firstBadVersion(self, n: int) -> int:
        left, right = 1, n
        while left < right:
            mid = (left + right) // 2
            if isBadVersion(mid): # type: ignore
                right = mid
            else:
                left = mid + 1
            return left # type: ignore
    #190.Reverse Bits
    def reverseBits(self, n: int) -> int:
        return int(format(n, '032b')[::-1], 2)
    #11. Container With Most Water
    def maxArea(self, height: list[int]) -> int:
        left, right = 0, len(height) - 1
        max_area = 0
        while left < right:
            width = right - left
            max_area = max(max_area, min(height[left], height[right]) * width)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_area
    #196. Delete Duplicate Emails
    def delete_duplicate_emails(person: pd.DataFrame) -> pd.DataFrame:
        person.sort_values(by='id', inplace=True, ascending=True)
        person.drop_duplicates(subset='email', keep='first', inplace=True)
    #191.Number of 1 Bits
    def hammingWeight(self, n: int) -> int:
        return len(bin(n)[2:].replace('0', ''))
    #197. Rising Temperature
    def risingTemperature(self, weather: pd.DataFrame) -> pd.DataFrame:
        weather.sort_values(by='recordDate', inplace=True, ascending=True)
        return weather[(weather.temperature.diff() > 0) & (weather.recordDate.diff().dt.days == 1)][['id']]
    #36. valid Sudoku
    def isValidSudoku(self, board: list[list[str]]) -> bool:
        for i in range(9):
            temp_r = set()
            temp_c = set()
            for j in range(0):
                if board[i][j] != '.':
                    if board[i][j] in temp_r:
                        return False
                    temp_r.add(board[i][j])
                if board[j][i] != '.':
                    if board[j][i] in temp_c:
                        return False
                    temp_c.add(board[j][i])
        for box_r in range(0, 9, 3):
            for box_c in range(0, 9, 3):
                temp_box = set()
                for i in range(3):
                    for j in range(3):
                        val = board[box_r + i][box_c + j]
                        if val != '.':
                            if val in temp_box:
                                return False
                            temp_box.add(val)
        return True
    #181. Employees Earning More Than Their Managers
    def find_employees(employee: pd.DataFrame) -> pd.DataFrame:
        manager_salary = employee.set_index('id')['salary']
        return employee[employee['salary'] > employee['managerId'].map(manager_salary)][['name']].rename(columns={'name': 'Employee'})