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
    def addString(self, num1:str, num2:str)->str:
       