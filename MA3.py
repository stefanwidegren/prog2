""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from numba import njit

# Exc1
def approximate_pi(n):
    inside = 0
    outside = 0

    inside_x = []
    inside_y = []
    outside_x = []
    outside_y = []

    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if x**2 + y**2 <= 1:
            inside += 1
            inside_x.append(x)
            inside_y.append(y)
        else:
            outside += 1
            outside_x.append(x)
            outside_y.append(y)

    pi_estimate = 4 * inside / (inside + outside)

    # Plot the points
    plt.scatter(inside_x, inside_y, color="red")
    plt.scatter(outside_x, outside_y, color="blue")

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    return pi_estimate 


# Exc2, approximation
def sphere_volume(n, d): 

    # we use list comprehension to generate n random points in d dimensions
    points = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]

    
    distance = lambda point: sum(x**2 for x in point) ** 0.5
    inside_sphere = filter(lambda point: distance(point) <= 1, points)

    pi_estimate = (len(list(inside_sphere)) / n) * (2 ** d)

    return pi_estimate

#Exc2, real value
def hypersphere_exact(n, d):

    return m.pi ** (d / 2) / m.gamma(d / 2 + 1)
    

#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:

    inside = 0

    for _ in range(n):
        squared_distance = 0.0

        for _ in range(d):
            coordinate = random.uniform(-1.0, 1.0)
            squared_distance += coordinate * coordinate

        if squared_distance <= 1.0:
            inside += 1

    return (inside / n) * (2 ** d)

#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel(n, d, np=10):
    processes = []

    with future.ProcessPoolExecutor(max_workers=np) as executor:
        # Each process estimates the volume using an equal share of the points.
        for _ in range(np):
            process = executor.submit(sphere_volume, n // np, d)
            processes.append(process)

        # Average the independent estimates returned by the processes.
        total = 0.0
        for process in processes:
            total += process.result()

    return total / np
    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    start = pc()
    sphere_volume_numba(n, d)
    stop = pc()
    print(f"Exc3: Numba time of {d} and {n}: {stop-start}")

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    sequential_time = stop - start
    print(f"Exc4: Sequential time of {d} and {n}: {sequential_time}")

    np = 2
    start = pc()
    sphere_volume_parallel(n, d, np)
    stop = pc()
    parallel_time = stop - start
    print(f"Exc4: Parallel time with {np} processes: {parallel_time}")
    
    

if __name__ == '__main__':
	main()
