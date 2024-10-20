import random

nums =  [random.randint(1,10000) for i in range(10)]
print("Правильный порядок: ", nums)
print("Инвертированный: ", list(reversed(nums)))

