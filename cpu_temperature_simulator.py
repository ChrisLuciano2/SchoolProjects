import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def cpu_temperature_ode(t, T, P_in_func, C, kappa, T_ambient):
    """
    Defines the ODE for the CPU temperature model.
    dT/dt = (1 / C) * ( P_in(t) - kappa * (T - T_ambient) )

    Args:
        t: Current time (s)
        T: Current temperature (°C)
        P_in_func: A function that returns the power input (W) at any time t.
        C: Thermal Capacitance (J/°C)
        kappa: Thermal Conductance (W/°C)
        T_ambient: Ambient Temperature (°C)

    Returns:
        dTdt: The rate of change of temperature at time t (°C/s)
    """
    # Get the current power input by calling the workload function
    current_power = P_in_func(t)
    # Calculate the rate of heat dissipation
    heat_loss = kappa * (T - T_ambient)
    # Apply the ODE
    dTdt = (1 / C) * (current_power - heat_loss)
    return dTdt

def main():
    print("ODE Solver for CPU Temperature Performance Metric")
    print("-------------------------------------------------")
    print("This model calculates the core temperature of a CPU over time based on its workload.")
    print("Units: Temperature [°C], Time [seconds], Power [Watts]")
    print()

    # Get user input for the ODE parameters (Thermal properties of the system)
    try:
        T_ambient = float(input("Enter the AMBIENT (room) temperature (°C): "))
        C = float(input("Enter the CPU's THERMAL CAPACITANCE (J/°C) [e.g., 5.0]: "))
        kappa = float(input("Enter the system's THERMAL CONDUCTANCE (W/°C) [e.g., 0.5]: "))
        t_max = float(input("Enter the total SIMULATION TIME (seconds): "))
    except ValueError:
        print("Error: Please enter valid numerical values.")
        return

    # --- Define the Workload (Power Input) as a function of time ---
    # This is a crucial part where the user defines the scenario.
    print("\nDefine the CPU workload (Power Input in Watts):")
    print("1. Constant workload")
    print("2. Periodic workload (simulates a game/application with cycles)")
    choice = input("Choose workload type (1 or 2): ")

    if choice == '1':
        P_constant = float(input("Enter the constant CPU power draw (W): "))
        # Define a simple function that always returns the constant power
        def P_in(t):
            return P_constant
        workload_description = f"Constant Workload: {P_constant} W"

    elif choice == '2':
        # A more complex function: base load + a sinusoidal load every 2 seconds
        P_base = float(input("Enter the base/idle power draw (W): "))
        P_peak = float(input("Enter the peak power draw (W): "))
        period = 2.0  # seconds per cycle
        def P_in(t):
            # Sinusoidal function oscillating between P_base and P_peak
            return P_base + (P_peak - P_base) * (np.sin(2 * np.pi * t / period) + 1)/2
        workload_description = f"Cyclical Workload: {P_base}-{P_peak} W"

    else:
        print("Invalid choice. Using constant workload of 50W.")
        def P_in(t):
            return 50.0
        workload_description = "Constant Workload: 50.0 W (default)"

    # Initial condition: start at ambient temperature
    T_initial = T_ambient

    # Time points for evaluation
    t_eval = np.linspace(0, t_max, 1000)

    # Solve the ODE
    # We must pass the P_in function and other parameters as arguments to the ODE function.
    solution = solve_ivp(cpu_temperature_ode,
                         [0, t_max],
                         [T_initial],
                         args=(P_in, C, kappa, T_ambient),
                         t_eval=t_eval,
                         method='RK45',
                         rtol=1e-8)  # High tolerance for a smooth physical result

    if not solution.success:
        print(f"ODE Solver failed: {solution.message}")
        return

    # Extract the solution
    time = solution.t
    temperature = solution.y[0]

    # --- Visualization ---
    plt.figure(figsize=(12, 7))

    # Plot 1: Temperature vs. Time
    plt.subplot(2, 1, 1) # 2 rows, 1 column, plot 1
    plt.plot(time, temperature, 'r-', linewidth=2, label='CPU Temperature')
    plt.axhline(y=T_ambient, color='b', linestyle='--', alpha=0.7, label=f'Ambient Temp ({T_ambient}°C)')
    
    # Calculate and plot theoretical maximum steady-state temperature
    # In steady state, dT/dt = 0 -> P_in = kappa * (T_ss - T_ambient)
    # For a constant workload, we can show the target. For dynamic, it's an estimate.
    avg_power = np.mean([P_in(t) for t in t_eval])
    T_steady_state = T_ambient + avg_power / kappa
    plt.axhline(y=T_steady_state, color='k', linestyle=':', label=f'Theoretical Max Steady-State (~{T_steady_state:.1f}°C)')
    
    # Critical temperature line (e.g., T_JunctionMax for many CPUs is ~100°C)
    critical_temp = 95.0
    plt.axhline(y=critical_temp, color='darkred', linestyle='-', alpha=0.4, label=f'Critical Temp ({critical_temp}°C)')
    
    plt.ylabel('Temperature (°C)')
    plt.title(f'CPU Temperature Performance Metric\n{workload_description}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(T_ambient - 5, max(max(temperature)*1.05, critical_temp * 1.1))

    # Plot 2: Power Input (Workload) vs. Time
    plt.subplot(2, 1, 2) # 2 rows, 1 column, plot 2
    power_values = [P_in(t) for t in t_eval]
    plt.plot(time, power_values, 'g-', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Power Input (Watts)')
    plt.title('CPU Workload (Power Draw)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(power_values) * 1.1)

    plt.tight_layout()
    plt.show()

    # --- Descriptive Output ---
    max_temp = np.max(temperature)
    final_temp = temperature[-1]
    print("\n--- Simulation Results ---")
    print(f"Maximum Temperature Reached: {max_temp:.2f} °C")
    print(f"Final Temperature: {final_temp:.2f} °C")
    print(f"Theoretical Steady-State Temperature: {T_steady_state:.2f} °C")

    # Performance analysis and warnings
    if max_temp > critical_temp:
        print(f"\n❌ WARNING: Maximum temperature exceeds critical limit of {critical_temp}°C.")
        print("This indicates insufficient cooling for the given workload. Performance will throttle.")
    elif max_temp > T_steady_state * 0.9:
        print(f"\n⚠️  NOTICE: System is operating near its thermal capacity.")
        print("The cooling solution is adequate but has little headroom.")
    else:
        print(f"\n✅ SUCCESS: Cooling solution is effective for this workload.")
        print("The CPU has significant thermal headroom.")

    # Error Estimate
    print(f"\nSolver Info: The solution has an estimated relative error per step < {solution.rtol:.1e}.")
    print(f"The solver took {solution.nfev} function evaluations to complete.")

if __name__ == "__main__":
    main()