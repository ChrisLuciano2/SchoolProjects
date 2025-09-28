Python ODE and Timing Examples
What This Does
This code shows two things:

Solves math equations (ODEs) using Python

Measures how long code takes to run

What You Need
bash
pip install numpy scipy matplotlib
How to Run
bash
python ode_example.py
What You'll See
Part 1: Equation Solving
Solves predator-prey population equations

Uses 3 different methods (RK45, Radau, BDF)

Shows they all get the same answer

Creates a graph of the results

Output:

Prey = 1.95, Predator = 1.03
(All three methods give same numbers)

Part 2: Timing Examples
Compares different ways to measure time

Shows bubble sort vs Python's built-in sort

Demonstrates which is faster

Output:

time.time(): 0.0054 seconds
time.perf_counter(): 0.0053 seconds
Bubble sort is 85x slower than built-in sort
Files Created
ode_comparison.png - Graph of the populations

Quick Start
Install packages above

Run the code

Watch the graph appear

See timing results in terminal

That's it! The code does the rest automatically.