# M/M/1 Queue Simulation

This project implements a discrete-event simulation of an M/M/1 queue system using Python. The M/M/1 queue is a classic model in queueing theory describing a single-server queue with Poisson arrivals and exponential service times.
The application features a graphical user interface (GUI) built with Tkinter and real-time plotting of queue length over time using matplotlib. It compares theoretical performance metrics with simulation results to validate the model and provides detailed per-customer statistics.

## Features

- Input parameters:  
  - Arrival rate (λ)  
  - Service rate (μ)  
  - Number of customers to simulate (n)
- Discrete-event simulation of arrivals, service start, and departures
- Calculation of waiting times, system times, and queue lengths at arrival
- Comparison of theoretical and simulated metrics such as:  
  - Average waiting time  
  - Average system time  
  - Average queue length  
  - Server utilization
- Visualization of queue length over time embedded in the GUI
- Detailed text output of customer-specific and aggregate queue statistics

## Usage

Run the mm1_queue_simulation.py script: python mm1_queue_simulation.py
Enter the desired input parameters for arrival rate, service rate, and number of customers, then click Run Simulation. The GUI will display results and a plot of queue length over time.
