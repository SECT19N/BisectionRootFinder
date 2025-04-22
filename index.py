import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


class BisectionMethodGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bisection Method Visualizer")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Variables
        self.function_var = tk.StringVar(value="x**2 - 4")
        self.a_var = tk.StringVar(value="0")
        self.b_var = tk.StringVar(value="3")
        self.tol_var = tk.StringVar(value="0.0001")
        self.max_iter_var = tk.StringVar(value="20")
        self.current_iter = 0
        self.iterations = []
        self.auto_play_speed = tk.DoubleVar(value=1.0)
        self.auto_play_active = False
        self.current_function = None  # Track current function
        self.current_a = None  # Track current a value
        self.current_b = None  # Track current b value

        # Create UI
        self.create_input_frame()
        self.create_control_frame()
        self.create_result_frame()
        self.create_plot_frame()

        # Initialize plots
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(2, 1, 1)  # Function plot
        self.ax2 = self.fig.add_subplot(2, 1, 2)  # Interval convergence

        # Embedding the matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

        # Initially disable step controls
        self.toggle_step_controls(False)

    def create_input_frame(self):
        # Frame for input parameters
        input_frame = ttk.LabelFrame(self.root, text="Input Parameters", padding=(10, 5))
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        # Function
        ttk.Label(input_frame, text="Function f(x):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.function_var, width=30).grid(row=0, column=1, sticky=tk.W, padx=5,
                                                                              pady=5)

        # a, b values
        ttk.Label(input_frame, text="Left endpoint (a):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.a_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(input_frame, text="Right endpoint (b):").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.b_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)

        # Tolerance and max iterations
        ttk.Label(input_frame, text="Tolerance:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.tol_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(input_frame, text="Max iterations:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.max_iter_var, width=10).grid(row=2, column=3, sticky=tk.W, padx=5,
                                                                              pady=5)

        # Start button
        start_button = ttk.Button(input_frame, text="Start Bisection Method", command=self.start_bisection)
        start_button.grid(row=3, column=0, columnspan=4, pady=10)

    def create_control_frame(self):
        # Frame for step controls
        self.control_frame = ttk.LabelFrame(self.root, text="Visualization Controls", padding=(10, 5))
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Step control buttons
        self.prev_button = ttk.Button(self.control_frame, text="◀ Previous Step", command=self.prev_step)
        self.prev_button.grid(row=0, column=0, padx=5, pady=5)

        self.next_button = ttk.Button(self.control_frame, text="Next Step ▶", command=self.next_step)
        self.next_button.grid(row=0, column=1, padx=5, pady=5)

        # Auto-play controls
        self.play_button = ttk.Button(self.control_frame, text="▶ Play", command=self.toggle_auto_play)
        self.play_button.grid(row=0, column=2, padx=5, pady=5)

        # Speed control
        ttk.Label(self.control_frame, text="Speed:").grid(row=0, column=3, padx=5, pady=5)
        speed_scale = ttk.Scale(self.control_frame, from_=0.1, to=3.0, variable=self.auto_play_speed,
                                orient=tk.HORIZONTAL, length=150)
        speed_scale.grid(row=0, column=4, padx=5, pady=5)

        # Reset button
        self.reset_button = ttk.Button(self.control_frame, text="Reset", command=self.reset)
        self.reset_button.grid(row=0, column=5, padx=5, pady=5)

        # Current iteration display
        self.iter_label = ttk.Label(self.control_frame, text="Iteration: 0/0")
        self.iter_label.grid(row=0, column=6, padx=20, pady=5)

    def create_result_frame(self):
        # Frame for displaying results
        self.result_frame = ttk.LabelFrame(self.root, text="Results", padding=(10, 5))
        self.result_frame.pack(fill=tk.X, padx=10, pady=5)

        # Text widget for displaying detailed information about current iteration
        self.result_text = tk.Text(self.result_frame, height=6, width=80, font=('Consolas', 10))
        self.result_text.pack(fill=tk.X, padx=5, pady=5)

        # Current iteration details
        self.details_frame = ttk.Frame(self.result_frame)
        self.details_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create labels for iteration details
        ttk.Label(self.details_frame, text="a:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.a_value_label = ttk.Label(self.details_frame, text="--")
        self.a_value_label.grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(self.details_frame, text="b:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.b_value_label = ttk.Label(self.details_frame, text="--")
        self.b_value_label.grid(row=0, column=3, sticky=tk.W, padx=5)

        ttk.Label(self.details_frame, text="c:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.c_value_label = ttk.Label(self.details_frame, text="--")
        self.c_value_label.grid(row=0, column=5, sticky=tk.W, padx=5)

        ttk.Label(self.details_frame, text="f(c):").grid(row=0, column=6, sticky=tk.W, padx=5)
        self.fc_value_label = ttk.Label(self.details_frame, text="--")
        self.fc_value_label.grid(row=0, column=7, sticky=tk.W, padx=5)

        ttk.Label(self.details_frame, text="Interval width:").grid(row=0, column=8, sticky=tk.W, padx=5)
        self.width_value_label = ttk.Label(self.details_frame, text="--")
        self.width_value_label.grid(row=0, column=9, sticky=tk.W, padx=5)

    def create_plot_frame(self):
        # Frame for matplotlib plots
        self.plot_frame = ttk.LabelFrame(self.root, text="Visualization", padding=(10, 5))
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def toggle_step_controls(self, enable):
        state = "normal" if enable else "disabled"
        self.prev_button["state"] = state
        self.next_button["state"] = state
        self.play_button["state"] = state
        self.reset_button["state"] = state

    def parse_expression(self, expr_str):
        """Parse a string expression into a SymPy expression, handling complex functions."""
        # Define symbolic function
        x = sp.symbols('x')
        try:
            # Replace common math functions that might not be recognized
            expr_str = expr_str.replace("^", "**")

            # Parse the expression
            return sp.sympify(expr_str)
        except Exception as e:
            raise ValueError(f"Error parsing function: {str(e)}")

    def safe_eval(self, f, x_val):
        """Safely evaluate function, handling potential complex results."""
        try:
            result = f(x_val)
            # If complex, take real part for visualization
            if isinstance(result, complex):
                return result.real
            return result
        except Exception as e:
            raise ValueError(f"Error evaluating function at x={x_val}: {str(e)}")

    def start_bisection(self):
        try:
            # Get inputs
            function_str = self.function_var.get()
            a = float(self.a_var.get())
            b = float(self.b_var.get())

            # Make sure a < b
            if a > b:
                a, b = b, a
                self.a_var.set(str(a))
                self.b_var.set(str(b))

            tol = float(self.tol_var.get())
            max_iter = int(self.max_iter_var.get())

            # Check if function is different from last run (force recomputation)
            if (function_str != self.current_function or
                    a != self.current_a or
                    b != self.current_b):

                # Store current values
                self.current_function = function_str
                self.current_a = a
                self.current_b = b

                # Parse and create the function
                function = self.parse_expression(function_str)
                x = sp.symbols('x')
                f = sp.lambdify(x, function, "numpy")

                # Check if f(a) and f(b) have opposite signs
                fa = self.safe_eval(f, a)
                fb = self.safe_eval(f, b)

                if fa * fb >= 0:
                    error_msg = f"Error: f(a) and f(b) must have opposite signs.\nf({a}) = {fa:.6f}, f({b}) = {fb:.6f}"
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, error_msg)
                    messagebox.showwarning("Invalid Interval",
                                           f"f(a) and f(b) must have opposite signs to guarantee a root in the interval. {fa * fb}")
                    return

                # Perform bisection method
                self.iterations = self.bisection_method(f, a, b, tol, max_iter)

                # Reset current iteration
                self.current_iter = 0

                # Update UI
                self.iter_label.config(text=f"Iteration: {self.current_iter}/{len(self.iterations) - 1}")
                self.toggle_step_controls(True)

                # Reset plots
                self.initial_a = a
                self.initial_b = b
                self.f = f
                self.function_str = function_str

                # Setup initial visualizations
                self.setup_plots()
                self.update_visualization()
                self.update_results()

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}\nPlease check your inputs.")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def bisection_method(self, f, a, b, tol, max_iter):
        """Implementation of the bisection method, returning all iterations."""
        iterations = [[a, b, None, None]]  # Initial interval, no midpoint yet

        for i in range(max_iter):
            c = (a + b) / 2
            fc = self.safe_eval(f, c)

            # Store current state
            iterations[-1][2] = c  # Add midpoint to current iteration
            iterations[-1][3] = fc  # Add f(c) to current iteration

            # Check termination
            if abs(fc) < tol or (b - a) / 2 < tol:
                break

            # Update interval
            fa = self.safe_eval(f, a)
            if fa * fc < 0:
                b = c
            else:
                a = c

            # Add new iteration with updated interval
            iterations.append([a, b, None, None])

        # Handle case where we have an incomplete last entry
        if iterations[-1][2] is None:
            iterations = iterations[:-1]

        return iterations

    def setup_plots(self):
        """Initialize the plot areas."""
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(2, 1, 1)  # Function plot
        self.ax2 = self.fig.add_subplot(2, 1, 2)  # Interval convergence

        # Determine plotting range
        padding = (self.initial_b - self.initial_a) * 0.2
        self.x_min = self.initial_a - padding
        self.x_max = self.initial_b + padding

        # Generate x and y values for function plot
        self.x_vals = np.linspace(self.x_min, self.x_max, 1000)
        self.y_vals = self.f(self.x_vals)

        # Plot function
        self.ax1.plot(self.x_vals, self.y_vals, 'b-', linewidth=2, label=f'f(x) = {self.function_str}')
        self.ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # Plot initial interval
        self.ax1.axvline(x=self.initial_a, color='g', linestyle='--', alpha=0.5, label='Initial interval')
        self.ax1.axvline(x=self.initial_b, color='g', linestyle='--', alpha=0.5)

        # Set up colors for iterations
        self.colors = plt.cm.viridis(np.linspace(0, 1, len(self.iterations)))

        # Set up the interval plot
        self.ax2.set_xlim(0, len(self.iterations) + 1)
        min_y = min(self.initial_a, min([it[0] for it in self.iterations])) - padding
        max_y = max(self.initial_b, max([it[1] for it in self.iterations])) + padding
        self.ax2.set_ylim(min_y, max_y)

        # Labels and titles
        self.ax1.set_xlabel('x', fontsize=12)
        self.ax1.set_ylabel('f(x)', fontsize=12)
        self.ax1.set_title(f'Bisection Method for f(x) = {self.function_str}', fontsize=14)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='best', fontsize=10)

        self.ax2.set_xlabel('Iteration', fontsize=12)
        self.ax2.set_ylabel('Interval [a, b]', fontsize=12)
        self.ax2.set_title('Interval Convergence', fontsize=14)
        self.ax2.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

    def update_visualization(self):
        """Update the plots for the current iteration."""
        if not self.iterations:
            return

        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Redraw the function
        self.ax1.plot(self.x_vals, self.y_vals, 'b-', linewidth=2, label=f'f(x) = {self.function_str}')

        # Add the zero line and initial interval lines
        self.ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax1.axvline(x=self.initial_a, color='g', linestyle='--', alpha=0.5, label='Initial interval')
        self.ax1.axvline(x=self.initial_b, color='g', linestyle='--', alpha=0.5)

        # Set up the interval plot again
        self.ax2.set_xlim(0, len(self.iterations) + 1)
        min_y = min(self.initial_a, min([it[0] for it in self.iterations])) - (self.initial_b - self.initial_a) * 0.1
        max_y = max(self.initial_b, max([it[1] for it in self.iterations])) + (self.initial_b - self.initial_a) * 0.1
        self.ax2.set_ylim(min_y, max_y)

        # Add labels for the interval plot
        self.ax2.set_xlabel('Iteration', fontsize=12)
        self.ax2.set_ylabel('Interval [a, b]', fontsize=12)
        self.ax2.set_title('Interval Convergence', fontsize=14)
        self.ax2.grid(True, alpha=0.3)

        # Loop through iterations up to current point
        for i in range(self.current_iter + 1):
            a, b, c, fc = self.iterations[i]
            color = self.colors[i]

            # Function plot: show interval and midpoint
            if i == self.current_iter:
                # Current iteration gets highlighted
                self.ax1.axvline(x=a, color='r', linestyle='-', linewidth=2, alpha=0.7,
                                 label=f'Iteration {i + 1}: [a, b]')
                self.ax1.axvline(x=b, color='r', linestyle='-', linewidth=2, alpha=0.7)
                if c is not None:
                    self.ax1.plot(c, fc, 'ro', markersize=8, label=f'Midpoint c = {c:.6f}')
                    self.ax1.plot([c, c], [0, fc], 'r--', alpha=0.7)
            else:
                # Past iterations are shown but less prominently
                alpha = 0.2 + (0.6 * i / max(1, self.current_iter))
                self.ax1.axvline(x=a, color=color, linestyle='-', alpha=alpha)
                self.ax1.axvline(x=b, color=color, linestyle='-', alpha=alpha)
                if c is not None:
                    self.ax1.plot(c, fc, 'o', color=color, markersize=6, alpha=alpha)

            # Interval visualization
            rect = Rectangle((i + 0.5, a), 0.8, b - a, color=color, alpha=0.7)
            self.ax2.add_patch(rect)
            self.ax2.plot([i + 0.9] * 2, [a, b], 'k-', alpha=0.5)
            if c is not None:
                self.ax2.plot(i + 0.9, c, 'ro', markersize=6)

        # Reset labels and titles for function plot (since we cleared the axes)
        self.ax1.set_xlabel('x', fontsize=12)
        self.ax1.set_ylabel('f(x)', fontsize=12)
        self.ax1.set_title(f'Bisection Method for f(x) = {self.function_str}', fontsize=14)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='best', fontsize=10)

        # Refresh the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def update_results(self):
        """Update the result text for the current iteration."""
        if not self.iterations or self.current_iter < 0 or self.current_iter >= len(self.iterations):
            return

        # Get current iteration data
        a, b, c, fc = self.iterations[self.current_iter]

        # Update iteration details labels
        self.a_value_label.config(text=f"{a:.6f}")
        self.b_value_label.config(text=f"{b:.6f}")

        if c is not None:
            self.c_value_label.config(text=f"{c:.6f}")
            self.fc_value_label.config(text=f"{fc:.6f}")
            self.width_value_label.config(text=f"{b - a:.6f}")
        else:
            self.c_value_label.config(text="--")
            self.fc_value_label.config(text="--")
            self.width_value_label.config(text=f"{b - a:.6f}")

        # Update the text widget with full details
        self.result_text.delete(1.0, tk.END)

        # Initial state
        if self.current_iter == 0:
            self.result_text.insert(tk.END, "Initial state:\n")
            self.result_text.insert(tk.END, f"Interval: [a, b] = [{a:.6f}, {b:.6f}]\n")
            self.result_text.insert(tk.END, f"f(a) = {self.f(a):.6f}, f(b) = {self.f(b):.6f}\n")
            self.result_text.insert(tk.END,
                                    "Since f(a) and f(b) have opposite signs, a root exists in this interval.\n")
            if c is not None:
                self.result_text.insert(tk.END, f"First midpoint: c = {c:.6f}, f(c) = {fc:.6f}\n")
        else:
            # Get previous iteration data for comparison
            prev_a, prev_b, prev_c, prev_fc = self.iterations[self.current_iter - 1]

            self.result_text.insert(tk.END, f"Iteration {self.current_iter}:\n")
            self.result_text.insert(tk.END, f"Previous interval: [{prev_a:.6f}, {prev_b:.6f}]\n")
            self.result_text.insert(tk.END, f"Previous midpoint: c = {prev_c:.6f}, f(c) = {prev_fc:.6f}\n")

            if prev_fc * self.f(prev_a) < 0:
                self.result_text.insert(tk.END, "Since f(a) and f(c) have opposite signs, the root is in [a, c].\n")
                self.result_text.insert(tk.END, f"New interval: [a, b] = [{a:.6f}, {b:.6f}]\n")
            else:
                self.result_text.insert(tk.END, "Since f(b) and f(c) have opposite signs, the root is in [c, b].\n")
                self.result_text.insert(tk.END, f"New interval: [a, b] = [{a:.6f}, {b:.6f}]\n")

            if c is not None:
                self.result_text.insert(tk.END, f"New midpoint: c = {c:.6f}, f(c) = {fc:.6f}\n")

                # Check for termination conditions
                if abs(fc) < float(self.tol_var.get()):
                    self.result_text.insert(tk.END,
                                            f"\n✓ |f(c)| = {abs(fc):.6f} < {float(self.tol_var.get()):.6f} (tolerance)\n")
                    self.result_text.insert(tk.END, "The root has been found within the specified tolerance!\n")
                elif (b - a) / 2 < float(self.tol_var.get()):
                    self.result_text.insert(tk.END,
                                            f"\n✓ Interval width = {(b - a) / 2:.6f} < {float(self.tol_var.get()):.6f} (tolerance)\n")
                    self.result_text.insert(tk.END, "The interval is smaller than the specified tolerance!\n")

    def next_step(self):
        """Move to the next iteration."""
        if self.current_iter < len(self.iterations) - 1:
            self.current_iter += 1
            self.iter_label.config(text=f"Iteration: {self.current_iter}/{len(self.iterations) - 1}")
            self.update_visualization()
            self.update_results()

    def prev_step(self):
        """Move to the previous iteration."""
        if self.current_iter > 0:
            self.current_iter -= 1
            self.iter_label.config(text=f"Iteration: {self.current_iter}/{len(self.iterations) - 1}")
            self.update_visualization()
            self.update_results()

    def toggle_auto_play(self):
        """Start or stop automatic iteration playback."""
        if self.auto_play_active:
            self.auto_play_active = False
            self.play_button.config(text="▶ Play")
        else:
            self.auto_play_active = True
            self.play_button.config(text="⏸ Pause")
            threading.Thread(target=self.auto_play, daemon=True).start()

    def auto_play(self):
        """Automatically step through all iterations."""
        while self.auto_play_active and self.current_iter < len(self.iterations) - 1:
            # Calculate delay based on speed
            delay = 1.0 / self.auto_play_speed.get()
            time.sleep(delay)

            # Update UI on the main thread
            self.root.after(0, self.next_step)

        # Reset play button when done
        if self.current_iter >= len(self.iterations) - 1:
            self.auto_play_active = False
            self.root.after(0, lambda: self.play_button.config(text="▶ Play"))

    def reset(self):
        """Reset to the first iteration."""
        self.current_iter = 0
        self.auto_play_active = False
        self.play_button.config(text="▶ Play")
        self.iter_label.config(text=f"Iteration: {self.current_iter}/{len(self.iterations) - 1}")
        self.update_visualization()
        self.update_results()


if __name__ == "__main__":
    root = tk.Tk()
    app = BisectionMethodGUI(root)
    root.mainloop()
