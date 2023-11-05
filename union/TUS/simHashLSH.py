from concurrent.futures import ProcessPoolExecutor

def square(x):
    return x * x

# 创建进程池，并使用 map 方法对 square 应用于列表中的每个元素
with ProcessPoolExecutor(max_workers=4) as executor:
    numbers = [1, 2, 3, 4]
    results = executor.map(square, numbers)

# 将结果转换为列表
squared_numbers = list(results)
print(squared_numbers)  # 输出: [1, 4, 9, 16, 25]
