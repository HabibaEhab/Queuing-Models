import tkinter as tk
from tkinter import scrolledtext, messagebox
import random
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class MM1QueueSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("M/M/1 Queue Simulation")
        self.root.geometry("1000x800")

        # Simulation variables
        self.arrival_rate = tk.DoubleVar(value=0.5)
        self.service_rate = tk.DoubleVar(value=1.0)
        self.num_customers = tk.IntVar(value=100)
        self.simulation_results = []

        self.create_widgets()

    def create_widgets(self):
        # Input frame
        input_frame = tk.LabelFrame(self.root, text="Simulation Parameters", padx=10, pady=10)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        # Arrival rate input
        tk.Label(input_frame, text="Arrival Rate (λ - customers/time unit):").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(input_frame, textvariable=self.arrival_rate).grid(row=0, column=1, sticky=tk.W)

        # Service rate input
        tk.Label(input_frame, text="Service Rate (μ - customers/time unit):").grid(row=1, column=0, sticky=tk.W)
        tk.Entry(input_frame, textvariable=self.service_rate).grid(row=1, column=1, sticky=tk.W)

        # Number of customers input
        tk.Label(input_frame, text="Number of Customers (n):").grid(row=2, column=0, sticky=tk.W)
        tk.Entry(input_frame, textvariable=self.num_customers).grid(row=2, column=1, sticky=tk.W)

        # Run button
        run_button = tk.Button(input_frame, text="Run Simulation", command=self.run_simulation)
        run_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Results frame
        results_frame = tk.LabelFrame(self.root, text="Results", padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Text output
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Theoretical vs Actual comparison
        comparison_frame = tk.Frame(results_frame)
        comparison_frame.pack(fill=tk.X, pady=5)

        tk.Label(comparison_frame, text="Theoretical vs Actual Comparison:", font=('Arial', 10, 'bold')).pack(
            anchor=tk.W)

        self.comparison_text = scrolledtext.ScrolledText(comparison_frame, height=8)
        self.comparison_text.pack(fill=tk.BOTH, expand=True)

        # Additional metrics frame
        metrics_frame = tk.Frame(results_frame)
        metrics_frame.pack(fill=tk.X, pady=5)

        tk.Label(metrics_frame, text="Additional Queue Metrics:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)

        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, height=8)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        # Plot frame
        plot_frame = tk.Frame(results_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_simulation(self):
        try:
            lambda_ = self.arrival_rate.get()
            mu = self.service_rate.get()
            n = self.num_customers.get()

            if lambda_ <= 0 or mu <= 0 or n <= 0:
                messagebox.showerror("Error", "All parameters must be positive numbers")
                return

            if lambda_ >= mu:
                messagebox.showerror("Error", "Arrival rate must be less than service rate (λ < μ) for stable queue")
                return

            # Run simulation
            self.simulation_results = self.simulate_mm1_queue(lambda_, mu, n)

            # Display results
            self.display_results()

            # Show theoretical vs actual comparison
            self.show_comparison(lambda_, mu)

            # Show additional metrics
            self.show_additional_metrics(lambda_, mu)

            # Plot queue length over time
            self.plot_queue_length()

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all parameters")

    def simulate_mm1_queue(self, lambda_, mu, n):
        results = []
        arrival_times = []
        service_times = []

        # Generate interarrival times (exponential distribution)
        interarrival_times = [random.expovariate(lambda_) for _ in range(n)]

        # Generate service times (exponential distribution)
        service_times = [random.expovariate(mu) for _ in range(n)]

        # Calculate arrival times
        arrival_times = [sum(interarrival_times[:i + 1]) for i in range(n)]

        # Initialize variables
        departure_times = [0] * n
        waiting_times = [0] * n
        system_times = [0] * n
        queue_lengths = [0] * n
        server_busy = False
        next_departure = float('inf')
        current_time = 0
        queue = []

        # Event-based simulation
        event_list = [(time, 'arrival', i) for i, time in enumerate(arrival_times)]
        event_list.sort()

        while event_list:
            current_time, event_type, customer = event_list.pop(0)

            if event_type == 'arrival':
                if not server_busy:
                    # Customer goes directly to service
                    server_busy = True
                    departure_time = current_time + service_times[customer]
                    next_departure = departure_time
                    departure_times[customer] = departure_time
                    waiting_times[customer] = 0
                    system_times[customer] = service_times[customer]
                    event_list.append((departure_time, 'departure', customer))
                    event_list.sort()
                else:
                    # Customer joins the queue
                    queue.append(customer)

                # Record queue length at arrival
                queue_lengths[customer] = len(queue)

            elif event_type == 'departure':
                if queue:
                    # Next customer from queue starts service
                    next_customer = queue.pop(0)
                    departure_time = current_time + service_times[next_customer]
                    next_departure = departure_time
                    departure_times[next_customer] = departure_time
                    waiting_times[next_customer] = current_time - arrival_times[next_customer]
                    system_times[next_customer] = departure_time - arrival_times[next_customer]
                    event_list.append((departure_time, 'departure', next_customer))
                    event_list.sort()
                else:
                    # No customers in queue, server becomes idle
                    server_busy = False
                    next_departure = float('inf')

        # Prepare results
        for i in range(n):
            results.append({
                'Customer': i + 1,
                'Arrival Time': arrival_times[i],
                'Service Time': service_times[i],
                'Departure Time': departure_times[i],
                'Waiting Time': waiting_times[i],
                'System Time': system_times[i],
                'Queue Length at Arrival': queue_lengths[i]
            })

        return results

    def display_results(self):
        self.results_text.delete(1.0, tk.END)

        if not self.simulation_results:
            return

        # Display header
        header = f"{'Customer':<10}{'Arrival':<10}{'Service':<10}{'Departure':<10}{'Wait':<10}{'System':<10}{'Q Length':<10}\n"
        self.results_text.insert(tk.END, header)
        self.results_text.insert(tk.END, "-" * 70 + "\n")

        # Display first 20 and last 20 customers (for brevity)
        for i in range(min(20, len(self.simulation_results))):
            cust = self.simulation_results[i]
            line = f"{cust['Customer']:<10}{cust['Arrival Time']:<10.2f}{cust['Service Time']:<10.2f}" \
                   f"{cust['Departure Time']:<10.2f}{cust['Waiting Time']:<10.2f}" \
                   f"{cust['System Time']:<10.2f}{cust['Queue Length at Arrival']:<10}\n"
            self.results_text.insert(tk.END, line)

        if len(self.simulation_results) > 40:
            self.results_text.insert(tk.END, "...\n")
            for i in range(max(20, len(self.simulation_results) - 20), len(self.simulation_results)):
                cust = self.simulation_results[i]
                line = f"{cust['Customer']:<10}{cust['Arrival Time']:<10.2f}{cust['Service Time']:<10.2f}" \
                       f"{cust['Departure Time']:<10.2f}{cust['Waiting Time']:<10.2f}" \
                       f"{cust['System Time']:<10.2f}{cust['Queue Length at Arrival']:<10}\n"
                self.results_text.insert(tk.END, line)

    def show_comparison(self, lambda_, mu):
        self.comparison_text.delete(1.0, tk.END)

        if not self.simulation_results:
            return

        # Calculate actual averages from simulation
        avg_waiting = sum(cust['Waiting Time'] for cust in self.simulation_results) / len(self.simulation_results)
        avg_system = sum(cust['System Time'] for cust in self.simulation_results) / len(self.simulation_results)
        avg_queue_length = sum(cust['Queue Length at Arrival'] for cust in self.simulation_results) / len(
            self.simulation_results)

        # Theoretical values
        rho = lambda_ / mu
        theoretical_avg_waiting = rho / (mu - lambda_)
        theoretical_avg_system = 1 / (mu - lambda_)
        theoretical_avg_queue_length = rho ** 2 / (1 - rho)

        comparison = (
            f"Theoretical vs Actual Comparison:\n"
            f"Parameter          Theoretical    Actual\n"
            f"----------------------------------------\n"
            f"Avg Waiting Time: {theoretical_avg_waiting:.4f}    {avg_waiting:.4f}\n"
            f"Avg System Time:  {theoretical_avg_system:.4f}    {avg_system:.4f}\n"
            f"Avg Queue Length: {theoretical_avg_queue_length:.4f}    {avg_queue_length:.4f}\n"
        )

        self.comparison_text.insert(tk.END, comparison)

    def show_additional_metrics(self, lambda_, mu):
        self.metrics_text.delete(1.0, tk.END)

        if not self.simulation_results:
            return

        rho = lambda_ / mu

        # Theoretical metrics
        P0 = 1 - rho  # Probability of 0 customers in system
        L = rho / (1 - rho)  # Average number of customers in system
        Pw = rho  # Probability server is busy (same as rho in M/M/1)

        # Calculate Pn for n=1 to 5
        Pn = {}
        for n in range(1, 6):
            Pn[n] = (1 - rho) * (rho ** n)

        # Simulation-based metrics
        total_time = self.simulation_results[-1]['Departure Time']
        idle_time = self.calculate_idle_time()
        sim_P0 = idle_time / total_time if total_time > 0 else 0
        sim_L = sum(cust['System Time'] for cust in self.simulation_results) / total_time if total_time > 0 else 0
        sim_Pw = 1 - sim_P0

        metrics = (
            f"Additional Queue Metrics (M/M/1):\n"
            f"Parameter          Theoretical    Simulation\n"
            f"----------------------------------------\n"
            f"Utilization (ρ):   {rho:.4f}\n"
            f"P₀ (0 customers):  {P0:.4f}        {sim_P0:.4f}\n"
            f"L (avg in system): {L:.4f}        {sim_L:.4f}\n"
            f"Pw (server busy):  {Pw:.4f}        {sim_Pw:.4f}\n\n"
            f"Probability of n customers in system (Pn):\n"
        )

        for n, prob in Pn.items():
            metrics += f"  P{n}: {prob:.4f}\n"

        self.metrics_text.insert(tk.END, metrics)

    def calculate_idle_time(self):
        if not self.simulation_results:
            return 0

        # Sort all events (arrivals and departures) by time
        events = []
        for cust in self.simulation_results:
            events.append((cust['Arrival Time'], 'arrival'))
            events.append((cust['Departure Time'], 'departure'))
        events.sort()

        idle_time = 0
        server_busy = False
        prev_time = 0

        for time, event_type in events:
            if not server_busy:
                idle_time += time - prev_time
            if event_type == 'arrival':
                server_busy = True
            else:  # departure
                # Check if there are any arrivals before the next departure
                server_busy = any(t > time and typ == 'arrival' for t, typ in events)
            prev_time = time

        return idle_time

    def plot_queue_length(self):
        if not self.simulation_results:
            return

        # Prepare data for plotting
        arrival_times = [cust['Arrival Time'] for cust in self.simulation_results]
        queue_lengths = [cust['Queue Length at Arrival'] for cust in self.simulation_results]

        # Create step plot
        self.ax.clear()
        self.ax.step(arrival_times, queue_lengths, where='post', label='Queue Length')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Queue Length')
        self.ax.set_title('Queue Length Over Time')
        self.ax.grid(True)
        self.ax.legend()

        self.canvas.draw()


def main():
    root = tk.Tk()
    app = MM1QueueSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()