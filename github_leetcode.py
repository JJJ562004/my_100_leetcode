class Solution:
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