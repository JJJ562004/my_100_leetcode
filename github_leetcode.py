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

