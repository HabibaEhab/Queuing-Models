import math
import random
from typing import Dict


# ------------------ Analytical Queues (Same as Before) ------------------

def mm1(lmbda: float, mu: float, n: int) -> Dict[str, float]:
    """M/M/1 queue model"""
    if lmbda >= mu:
        raise ValueError("λ must be less than μ for M/M/1 queue stability")

    rho = lmbda / mu
    P0 = 1 - rho
    Pn = (1 - rho) * (rho ** n)
    L = rho / (1 - rho)
    Lq = rho ** 2 / (1 - rho)
    W = 1 / (mu - lmbda)
    Wq = lmbda / (mu * (mu - lmbda))
    Pw = rho

    return {
        'model': 'M/M/1 - Basic single-server queue',
        'lambda': lmbda,
        'mu': mu,
        'P0': P0,
        'rho': rho,
        'L': L,
        'Lq': Lq,
        'W': W,
        'Wq': Wq,
        'Pw': Pw,
        'Pn': Pn
    }


def mmk(lmbda: float, mu: float, k: int, n: int) -> Dict[str, float]:
    """M/M/k queue model"""
    rho = lmbda / (k * mu)
    if rho >= 1:
        raise ValueError("λ must be less than k*μ for M/M/k queue stability")

    sum_terms = sum((lmbda / mu) ** n / math.factorial(n) for n in range(k))
    last_term = ((lmbda / mu) ** k / math.factorial(k)) * (1 / (1 - rho))
    P0 = 1 / (sum_terms + last_term)

    if n <= k:
        Pn = ((lmbda / mu) ** n / math.factorial(n)) * P0
    else:
        Pn = ((lmbda / mu) ** n / (math.factorial(k) * k ** (n - k))) * P0

    Lq = (P0 * (lmbda / mu) ** k * rho) / (math.factorial(k) * (1 - rho) ** 2)
    L = Lq + lmbda / mu
    W = L / lmbda
    Wq = Lq / lmbda
    Pw = (1 / math.factorial(k)) * ((lmbda / mu) ** k) * (k * mu / (k * mu - lmbda)) * P0

    return {
        'model': 'M/M/k - Multi-server queue with k servers',
        'lambda': lmbda,
        'mu': mu,
        'servers': k,
        'P0': P0,
        'rho': rho,
        'L': L,
        'Lq': Lq,
        'W': W,
        'Wq': Wq,
        'Pw': Pw,
        'Pn': Pn
    }


def mm1k(lmbda: float, mu: float, K: int, n: int) -> Dict[str, float]:
    """M/M/1/K queue model (finite buffer)"""
    rho = lmbda / mu

    if rho == 1:
        P0 = 1 / (K + 1)
    else:
        P0 = (1 - rho) / (1 - rho ** (K + 1))

    if n <= K:
        Pn = (rho ** n) * P0
    else:
        Pn = 0

    if rho == 1:
        L = K / 2
    else:
        L = rho * (1 - (K + 1) * rho ** K + K * rho ** (K + 1)) / ((1 - rho) * (1 - rho ** (K + 1)))

    Lq = max(L - (1 - P0), 0)
    lambda_eff = lmbda * (1 - (rho ** K * P0 if rho != 1 else P0))
    W = L / lambda_eff if lambda_eff > 0 else float('inf')
    Wq = Lq / lambda_eff if lambda_eff > 0 else float('inf')
    Pw = 1 - P0

    return {
        'model': 'M/M/1/K - Single-server queue with finite buffer size K',
        'lambda': lmbda,
        'mu': mu,
        'buffer_size': K,
        'P0': P0,
        'rho': rho,
        'L': L,
        'Lq': Lq,
        'W': W,
        'Wq': Wq,
        'Pw': Pw,
        'Pn': Pn
    }


def mminf(lmbda: float, mu: float, n: int) -> Dict[str, float]:
    """M/M/∞ queue model (infinite servers)"""
    rho = lmbda / mu

    P0 = math.exp(-rho)
    Pn = (math.exp(-rho) * (rho ** n)) / math.factorial(n)
    L = rho
    Lq = 0
    W = 1 / mu
    Wq = 0
    Pw = 0

    return {
        'model': 'M/M/∞ - Infinite servers model',
        'lambda': lmbda,
        'mu': mu,
        'P0': P0,
        'rho': rho,
        'L': L,
        'Lq': Lq,
        'W': W,
        'Wq': Wq,
        'Pw': Pw,
        'Pn': Pn
    }


def mm1m(lmbda: float, mu: float, m: int, n: int) -> Dict[str, float]:
    """M/M/1/m queue model (finite population)"""
    r = lmbda / mu
    P = [0] * (m + 1)
    P[0] = 1.0

    for i in range(1, m + 1):
        P[i] = P[i - 1] * ((m - (i - 1)) * lmbda) / (i * mu)

    P0 = 1 / sum(P)
    P = [p * P0 for p in P]

    L = sum(i * P[i] for i in range(m + 1))
    lambda_eff = lmbda * sum((m - i) * P[i] for i in range(m + 1))
    W = L / lambda_eff if lambda_eff != 0 else float('inf')
    Lq = sum(max(i - 1, 0) * P[i] for i in range(m + 1))
    Wq = Lq / lambda_eff if lambda_eff != 0 else float('inf')
    Pw = 1 - P[0]

    return {
        'model': 'M/M/1/m - Finite population model with m customers',
        'lambda': lmbda,
        'mu': mu,
        'population': m,
        'P0': P[0],
        'rho': r,
        'L': L,
        'Lq': Lq,
        'W': W,
        'Wq': Wq,
        'Pw': Pw,
        'Pn': P[n] if n <= m else 0
    }


# ------------------ Simulation (Same as Before) ------------------

def simulate_mm1(lmbda: float, mu: float, simulation_time: float, n: int) -> Dict[str, float]:
    """Simulation of M/M/1 queue"""
    clock = 0
    queue = []
    arrival_time = random.expovariate(lmbda)
    service_end_time = float('inf')
    wait_times = []
    system_times = []
    queue_lengths = []
    total_customers = 0
    state_counts = [0] * (n + 2)  # Track system states up to n+1

    while clock < simulation_time:
        if arrival_time < service_end_time:
            clock = arrival_time
            queue.append(clock)
            arrival_time = clock + random.expovariate(lmbda)
            if len(queue) == 1:
                service_end_time = clock + random.expovariate(mu)
        else:
            clock = service_end_time
            arrival = queue.pop(0)
            wait_times.append(clock - arrival)
            system_times.append(clock - arrival)
            total_customers += 1
            if queue:
                service_end_time = clock + random.expovariate(mu)
            else:
                service_end_time = float('inf')

        # Record current system state
        current_state = len(queue)
        if current_state <= n + 1:
            state_counts[current_state] += (min(arrival_time, service_end_time) - clock) if clock > 0 else 0

        queue_lengths.append(len(queue))

    total_time = clock
    state_probabilities = [count / total_time for count in state_counts]

    avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
    avg_system = sum(system_times) / len(system_times) if system_times else 0
    avg_queue_length = sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0

    rho = lmbda / mu
    P0 = max(0, 1 - rho)
    Pn = state_probabilities[n] if n < len(state_probabilities) else 0

    return {
        'model': 'M/M/1 Simulation - Basic single-server queue',
        'lambda': lmbda,
        'mu': mu,
        'P0': P0,
        'rho': rho,
        'L': avg_system * lmbda,
        'Lq': avg_queue_length,
        'W': avg_system,
        'Wq': avg_wait,
        'Pw': rho,
        'Pn': Pn,
        'simulation_time': simulation_time,
        'total_customers': total_customers,
        'avg_queue_length': avg_queue_length,
        'avg_wait': avg_wait,
        'avg_system': avg_system
    }


# ------------------ Output Formatter (Same as Before) ------------------

def format_output(results: Dict[str, float]):
    """Format output exactly as specified in the requirements"""
    print("\n* ***Outputs:**")
    print(f"1. Type of the queuing model related to the system: {results['model']}")
    print(f"2. λ = {results['lambda']}, μ = {results['mu']}")
    # Print additional parameters if they exist
    if 'servers' in results:
        print(f"   - k (servers) = {results['servers']}")
    if 'buffer_size' in results:
        print(f"   - K (buffer size) = {results['buffer_size']}")
    if 'population' in results:
        print(f"   - m (population) = {results['population']}")
    print("3. ALL the performance measures:")
    print(f"   - P₀ = {results['P0']:.4f}")
    print(f"   - ρ = {results['rho']:.4f}")
    print(f"   - L = {results['L']:.4f}")
    print(f"   - Lq = {results['Lq']:.4f}")
    print(f"   - W = {results['W']:.4f}")
    print(f"   - Wq = {results['Wq']:.4f}")
    print(f"   - Pw = {results['Pw']:.4f}")
    print(f"4. P₀ = {results['P0']:.4f}, Pn = {results['Pn']:.4f}")

    # Add simulation-specific output if present
    if 'simulation_time' in results:
        print("\nSimulation Details:")
        print(f"   - Simulation time: {results['simulation_time']}")
        print(f"   - Total customers served: {results['total_customers']}")
        print(f"   - Average queue length observed: {results['avg_queue_length']:.4f}")
        print(f"   - Average wait time observed (Wq): {results['avg_wait']:.4f}")
        print(f"   - Average system time observed (W): {results['avg_system']:.4f}")

    print("=" * 50)


# ------------------ Automatic Model Selection ------------------

def auto_select_model(lmbda: float, mu: float, n: int, k: int = None, K: int = None, m: int = None,
                      simulation: bool = False, sim_time: float = 1000) -> Dict[str, float]:
    """Automatically select the queuing model based on inputs."""
    if simulation:
        return simulate_mm1(lmbda, mu, sim_time, n)
    elif m is not None:
        return mm1m(lmbda, mu, m, n)
    elif K is not None:
        return mm1k(lmbda, mu, K, n)
    elif k is not None and k > 1:
        return mmk(lmbda, mu, k, n)
    else:
        return mm1(lmbda, mu, n)


# ------------------ User Input (Now Simpler) ------------------

def get_user_input():
    """Get input parameters from user (no model selection step)."""
    print("Queuing System Analyzer")
    print("Enter the parameters of your system:")

    lmbda = float(input("Arrival rate (λ): "))
    mu = float(input("Service rate (μ): "))
    n = int(input("Number of customers to calculate Pn for (n): "))

    # Optional parameters (default to None if not provided)
    k = None
    K = None
    m = None
    simulation = False
    sim_time = 1000

    # Check for additional parameters
    has_servers = input("Number of servers (k, press Enter if 1): ")
    if has_servers:
        k = int(has_servers)

    has_buffer = input("Buffer size (K, press Enter if infinite): ")
    if has_buffer:
        K = int(has_buffer)

    has_population = input("Population size (m, press Enter if infinite): ")
    if has_population:
        m = int(has_population)

    # Auto-select model
    return auto_select_model(lmbda, mu, n, k, K, m, simulation, sim_time)


# ------------------ Main Program ------------------

if __name__ == "__main__":
    try:
        results = get_user_input()
        format_output(results)
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")