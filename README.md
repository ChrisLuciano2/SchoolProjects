# CPU Temperature Simulator

Before running this program, ensure you have Python 3.7 or higher installed on your system.

## Installation & Setup

1.  Download the Script
    - Save the `cpu_temperature_simulator.py` file to a directory on your computer.

2.  Install Required Libraries
    - Open a terminal (Command Prompt, PowerShell, or shell).
    - Run the following command to install the necessary Python packages:

      ```bash
      pip install numpy scipy matplotlib
      ```

## How to Run the Program

1.  Open a Terminal
    - Navigate to the directory where you saved the Python script.

    ```bash
    cd path/to/your/directory
    ```

2.  Execute the Script
    - Run the program using Python:

    ```bash
    python cpu_temperature_simulator.py
    ```
    *you may need to use `python3` or `py` instead of `python`.*

3.  Follow the On-Screen Prompts
    - Enter the thermal properties of your system when prompted:
        - AMBIENT temperature (°C)
        - THERMAL CAPACITANCE (J/°C) (e.g., 5.0)
        - THERMAL CONDUCTANCE (W/°C) (e.g., 0.5)
        - SIMULATION TIME (seconds)
    - Choose a workload type (constant or cyclical) and provide the corresponding power values (in Watts).

4.  View the Results
    - The program will generate a graph displaying the CPU temperature and power input over time.
    - Close the graph window to see the numerical results and performance analysis printed in the terminal.

## Example Usage
Enter the AMBIENT (room) temperature (°C): 22
Enter the CPU's THERMAL CAPACITANCE (J/°C) [e.g., 5.0]: 5.0
Enter the system's THERMAL CONDUCTANCE (W/°C) [e.g., 0.5]: 0.5
Enter the total SIMULATION TIME (seconds): 30

Define the CPU workload (Power Input in Watts): 100

1.Constant workload

2.Periodic workload (simulates a game/application with cycles)
Choose workload type (1 or 2): 2
Enter the base/idle power draw (W): 20
Enter the peak power draw (W): 100