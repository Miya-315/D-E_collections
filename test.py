# num =[2,3,1,4],stand = [0,3],则输出为 [2,1,3,4]

def sort_nums(nums, stand):

    n = len(nums)
    stand_set = set(stand)
    
    others = [nums[i] for i in range(n) if i not in stand_set]
    others.sort()
    
    res = []
    it = iter(others)
    for i in range(n):
        if i in stand_set:
            res.append(nums[i])
        else:
            res.append(next(it))
    return res

nums = [2, 3, 1, 4]
stand = [0, 3]
print(sort_nums(nums, stand))  # 期望 [2,1,3,4]





