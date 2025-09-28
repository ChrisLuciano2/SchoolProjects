# ode_and_timing.py
# Combined demonstration of ODE solving and timing methods in Python

print("ODE SOLVING AND TIMING METHODS")

# Part 1: ODE Solving Packages in Python
print("\nPART 1: ODE SOLVING PACKAGES")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    
    print("✓ Successfully imported ODE solving packages")
    
    # Define a system of ODEs: Lotka-Volterra predator-prey model
    def predator_prey(t, y, alpha=1.1, beta=0.4, gamma=0.4, delta=0.1):
        prey, predator = y
        dprey_dt = alpha * prey - beta * prey * predator
        dpredator_dt = delta * prey * predator - gamma * predator
        return [dprey_dt, dpredator_dt]
    
    # Initial conditions and time span
    y0 = [10, 5]  # Initial populations: [prey, predator]
    t_span = (0, 50)
    t_eval = np.linspace(0, 50, 1000)
    
    print("Solving Lotka-Volterra predator-prey equations...")
    
    # Solve using different methods and compare
    methods = ['RK45', 'Radau', 'BDF']
    solutions = {}
    
    for method in methods:
        sol = solve_ivp(predator_prey, t_span, y0, method=method, 
                       t_eval=t_eval, rtol=1e-6)
        solutions[method] = sol
        print(f"✓ {method}: Completed with {sol.y.shape[1]} time points")
    
    # Display final results
    print("\nFinal population values (t=50):")
    for method in methods:
        final_prey = solutions[method].y[0, -1]
        final_predator = solutions[method].y[1, -1]
        print(f"  {method}: Prey = {final_prey:.2f}, Predator = {final_predator:.2f}")
    
    # Create a plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for method in methods:
        plt.plot(solutions[method].t, solutions[method].y[0], label=f'{method} - Prey')
    plt.title('Prey Population Over Time')
    plt.xlabel('Time')
    plt.ylabel('Prey Population')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for method in methods:
        plt.plot(solutions[method].t, solutions[method].y[1], label=f'{method} - Predator')
    plt.title('Predator Population Over Time')
    plt.xlabel('Time')
    plt.ylabel('Predator Population')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ode_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ ODE solutions plotted and saved as 'ode_comparison.png'")
    
except ImportError as e:
    print(f"❌ Error importing ODE packages: {e}")
    print("Please install required packages: pip install numpy scipy matplotlib")

# Part 2: Timing Methods in Python
print("\nPART 2: COMPUTING TIME MEASUREMENT TECHNIQUES")

import time
import timeit
import random

# Example 1: Compare different timing functions
print("\n1. COMPARING TIMING FUNCTIONS:")

def computational_task(n):
    """An expensive computational task"""
    result = 0
    for i in range(n):
        result += i ** 2 + i ** 0.5
    return result

n = 50000

# Method 1: time.time()
start_time = time.time()
result1 = computational_task(n)
end_time = time.time()
time_time = end_time - start_time

# Method 2: time.perf_counter()
start_perf = time.perf_counter()
result2 = computational_task(n)
end_perf = time.perf_counter()
time_perf = end_perf - start_perf

# Method 3: time.process_time()
start_process = time.process_time()
result3 = computational_task(n)
end_process = time.process_time()
time_process = end_process - start_process

print(f"time.time():        {time_time:.6f} seconds")
print(f"time.perf_counter(): {time_perf:.6f} seconds")
print(f"time.process_time(): {time_process:.6f} seconds")

# Example 2: Using timeit for reliable benchmarking
print("\n2. RELIABLE BENCHMARKING WITH timeit:")

def matrix_operations():
    """Various matrix operations"""
    A = np.random.rand(100, 100)
    B = np.random.rand(100, 100)
    C = np.dot(A, B)  # Matrix multiplication
    det = np.linalg.det(C)  # Determinant
    return det

# Single execution time
single_time = timeit.timeit(matrix_operations, number=1)
print(f"Single execution: {single_time:.6f} seconds")

# Multiple executions with statistics
repeat_times = timeit.repeat(matrix_operations, number=1, repeat=5)
print(f"Five executions: {repeat_times}")
print(f"Average: {np.mean(repeat_times):.6f} ± {np.std(repeat_times):.6f} seconds")

# Example 3: Context manager for timing
print("\n3. CONTEXT MANAGER FOR CLEAN TIMING:")

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        print(f"  Execution time: {self.interval:.6f} seconds")

print("Timing list operations:")
with Timer():
    large_list = [random.randint(1, 1000) for _ in range(100000)]
    sorted_list = sorted(large_list)
    unique_elements = set(sorted_list)

# Example 4: Algorithm comparison
print("\n4. ALGORITHM PERFORMANCE COMPARISON:")

def bubble_sort(arr):
    """Bubble sort algorithm - O(n²) complexity"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def builtin_sort(arr):
    """Python's built-in sort - O(n log n) complexity"""
    return sorted(arr)

# Test data
test_data = [random.randint(0, 10000) for _ in range(2000)]

# Time both algorithms
bubble_time = timeit.timeit(lambda: bubble_sort(test_data.copy()), number=3)
builtin_time = timeit.timeit(lambda: builtin_sort(test_data.copy()), number=3)

print(f"Bubble sort (3 runs): {bubble_time:.6f} seconds")
print(f"Built-in sort (3 runs): {builtin_time:.6f} seconds")
print(f"Performance ratio: {bubble_time/builtin_time:.1f}x faster")

print("\nDEMONSTRATION COMPLETED SUCCESSFULLY!")