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

- Run the mm1_queue_simulation.py
- Enter the desired input parameters for arrival rate, service rate, and number of customers, then click Run Simulation. The GUI will display results and a plot of queue length over time.



--

# Queuing System Analyzer and Simulator

This Python project implements analytical models and a discrete-event simulation for classical queuing systems including M/M/1, M/M/k, M/M/1/K, M/M/1/m, and M/M/∞ queues.

## Features

- Analytical computation of performance metrics for:
  - M/M/1 (single server, infinite buffer)
  - M/M/k (multiple servers)
  - M/M/1/K (finite buffer)
  - M/M/1/m (finite population)
  - M/M/∞ (infinite servers)
- Discrete-event simulation of the M/M/1 queue
- Automatic model selection based on input parameters
- Interactive user input interface
- Detailed formatted output of key performance measures:
  - Probability of zero customers (P₀)
  - Utilization (ρ)
  - Average number in system (L)
  - Average number in queue (Lq)
  - Average waiting time in system (W)
  - Average waiting time in queue (Wq)
  - Probability server is busy (Pw)
  - Probability of n customers in system (Pn)

## Usage
- Run the program
- You will be prompted to enter the following parameters:
- Arrival rate (λ)
- Service rate (μ)
- Number of customers to calculate Pₙ for (n)
- Optional: Number of servers (k), buffer size (K), population size (m)
- The program automatically selects the appropriate queuing model and outputs performance metrics.


