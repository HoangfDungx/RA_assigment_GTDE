import numpy as np
import components

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

L = 100000

class MILPOptimizer:
    def __init__(self, input_set) -> None:
        """
            Initialize optimizer model
        """
        self.model = gp.Model("AGVTrans")

        M = input_set['input_data']['num_machine']
        J = input_set['input_data']['num_job']
        V = input_set['input_data']['num_AGV']
        """
            agv_set[i] is AGV i
        """
        self.agv_set = [0]
        for k in range(1, V + 1):
            self.agv_set.append(components.AGV(input_set['input_data']['AGV_color'][k - 1]))

        """
            operation_set[i][j] is operation of job j in machine i
        """
        self.operation_set = [[]]   # Ignore index 0
        for i in range(1, M + 1):
            new_machine = [None]    # Ignore index 0
            for j in range(1, J + 1):
                print(f"- New operation of job {j} in machine {i} with process time: \
                      {input_set['machine_process_time'][i - 1]}")
                new_machine.append(components.Operation(input_set['machine_process_time'][i - 1], \
                                                        self.model, machine_id=i, job_id=j))
            self.operation_set.append(new_machine)
        
        """
            transportation_set[i][j] is transportation of job j to machine i
        """
        self.transportation_set = [[],[]]   # Ignore index 0 and machine 1
        for i in range(2, M + 1):
            new_machine = [None]    # Ignore index 0
            for j in range(1, J + 1):
                print(f"- New transportation of job {j} to machine {i} with transportation time: \
                      {input_set['machine_layout_mtx'][i - 2][i - 1]}")
                new_machine.append(components.Transportation(V, \
                                                             input_set['machine_layout_mtx'][i - 2][i - 1], \
                                                             self.model, machine_id=i, job_id=j))
            self.transportation_set.append(new_machine)

        """
            travel_time_mtx[i][j] is travel time from i to j
        """
        self.travel_time_mtx = [[]]                                        # Ignore index 0 each column
        for i in range(len(input_set['machine_layout_mtx'])):
            new_row = [0.0]                                               # Ignore index 0 each row
            for j in range(len(input_set['machine_layout_mtx'][i])):
                new_row.append(input_set['machine_layout_mtx'][i][j])
            self.travel_time_mtx.append(new_row)

        print("[MILPOptimizer] travel_time_mtx: ")
        print(self.travel_time_mtx)

        """
            operation_time[i] is processing time of machine i   
        """
        self.operation_time = [0]
        for i in range(M):
            self.operation_time.append(input_set['machine_process_time'][i])

        print("[MILPOptimizer] operation_time: ")
        print(self.operation_time)

        """
            operation_rules_mtx indicates whether operation ji precedes operation j'i' or not
        """
        self.operation_rules_mtx = [[]]
        for i in range(1, M + 1):
            orm_2 = [[]]
            for j in range(1, J + 1):
                """
                    Loop through all operation twice
                """
                orm_3 = [[]]
                for i_ in range(1, M + 1):
                    orm_4 = [0]
                    for j_ in range(1, J + 1):
                        if j == j_:
                            """
                                if j == j_ as also same job, it is known and not a variable
                            """
                            orm_4.append(1 if i < i_ else 0)
                        else:
                            orm_4.append(self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY, \
                                                           name="y^({j}|{i})/({j_}|{i_})"))
                    
                    orm_3.append(orm_4)
                orm_2.append(orm_3)
            self.operation_rules_mtx.append(orm_2)
                            
        """
            Initialize constraints between operation and transportation
        """
        # Constraint 3, 4
        for i in range(2, M + 1):
            for j in range(1, J + 1):
                new_constr3 = (self.operation_set[i][j].start_time >= \
                              self.transportation_set[i][j].start_time + self.transportation_set[i][j].time)
                self.model.addConstr(new_constr3)

                new_constr4 = (self.transportation_set[i][j].start_time >= \
                               self.operation_set[i - 1][j].start_time + self.operation_set[i - 1][j].time)
                self.model.addConstr(new_constr4)

        # Constraint 5
        for i in range(1, M):
            for j in range(1, J + 1):
                for j_ in range(1, J + 1):
                    if j == j_:
                        continue

                    new_constr5 = (self.operation_set[i][j_].start_time + L * (1 - self.operation_rules_mtx[i][j][i][j_]) >= \
                                   self.transportation_set[i + 1][j].start_time)
                    self.model.addConstr(new_constr5)

        # Constraint 6 for the last machine without transportation out
        for j in range(1, J + 1):
            for j_ in range(1, J + 1):
                if j == j_:
                    continue

                new_constr6 = (self.operation_set[M][j_].start_time + L * (1 - self.operation_rules_mtx[M][j][M][j_]) >= \
                                self.operation_set[M][j].start_time + self.operation_set[M][j].time)
                self.model.addConstr(new_constr6)

        # Operation constrain 8: order between two operations
        for i in range(1, M + 1):
            for j in range(1, J + 1):
                for i_ in range(1, M + 1):
                    for j_ in range(1, J + 1):
                        if j >= j_:
                            continue

                        new_constr8 = (self.operation_rules_mtx[i][j][i_][j_] + self.operation_rules_mtx[i_][j_][i][j] == 1)
                        self.model.addConstr(new_constr8)

        """
            AGV constraints
        """
        # Constraint 7: Which AGV carry out which operation
        for i in range(2, M + 1):
            for j in range(1, J + 1):
                new_constr7 = (sum(self.transportation_set[i][j].agv_state) == 1)
                self.model.addConstr(new_constr7)

        # Constraint 9: AGV and transportation
        for i in range(1, M):
            for j in range(1, J + 1):
                for i_ in range(2, M + 1):
                    for j_ in range(1, J + 1):
                        for k in range(1, V + 1):
                            new_constr9 = (
                                self.transportation_set[i + 1][j].start_time + \
                                    L * (3 - self.transportation_set[i + 1][j].agv_state[k]
                                         - self.transportation_set[i_][j_].agv_state[k]
                                         - self.operation_rules_mtx[i_][j_][i + 1][j]) >= \
                                self.operation_set[i_][j_].start_time + self.travel_time_mtx[i_][i]
                            )
                            self.model.addConstr(new_constr9)

        """
            Objective function
        """
        # Add Cmax as a variable
        self.Cmax = self.model.addVar(lb=0.0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="makespan")
        # Add constraints: Cmax is always greater the finishing times of operations in machine M
        for j in range(1, J + 1):
            new_makespan_constraint = (self.Cmax >= self.operation_set[M][j].start_time + self.operation_set[M][j].time)
            self.model.addConstr(new_makespan_constraint)

        self.model.setObjective(self.Cmax, GRB.MINIMIZE)


    def solve(self):
        self.model.optimize()
        """
            print result
        """
        print(f"[MILPOptimizer] Optimization result: Make span: {self.Cmax.X}")

        """
            Most thing is well done, but we need to arrange empty trip for AGVs
            Firstly, each AGV need to sort their transportation in order
            Then, the empty trip is their last operation starting time to the next trip
        """
        # Loop through all transportation and add into agv
        for i in range(2, len(self.transportation_set)):
            for j in range(1, len(self.transportation_set[i])):
                k = self.transportation_set[i][j].getAGV()
                self.agv_set[k].add_transportation(i, j, self.transportation_set[i][j].start_time.X)

        for k in range(1, len(self.agv_set)):
            self.agv_set[k].schedule_empty_trip()


    def visualize_result(self, mode):
        """
            Create a chart to indicate manufacturing process
        """
        # Define the number of machines and time range
        M = len(self.operation_set)
        # t = round(self.Cmax.X)
        t = round(self.Cmax.X)

        # Create lists for machine numbers and time points
        machines = list(range(1, M + 1))
        time_points = list(range(t))

        # Create the plot
        plt.figure(figsize=(20, 10))

        # Draw vertical dashed lines for each machine
        for machine in machines:
            plt.axhline(y=machine, color='gray', linestyle='--', alpha=0.3)

        self.plot(mode)

        # Set labels and title
        plt.xlabel("Time")
        plt.ylabel("Machine")
        plt.title("Manufacturing schedule chart")

        # Show the legend
        self.create_legend()

        # Show the plot
        plt.show()


    def plot(self, mode):
        half_bar_width = 0.1
        half_space_width = 0.2

        self.visualize_operation_m1()

        for k in range(1, len(self.agv_set)):
            # Plot operation
            for transportation in self.agv_set[k].transportation_set:
                i = transportation[0]
                j = transportation[1]
                a = self.operation_set[i][j].start_time.X
                b = self.operation_set[i][j].start_time.X + self.operation_set[i][j].time
                plt.fill_betweenx(y=[i - half_bar_width, i + half_bar_width], \
                                  x1=a, x2=b, color=self.agv_set[k].color, alpha=0.3, edgecolor='black')
                plt.text((a + b) / 2, i, f"Job {j}",ha="center", va="center", fontsize=10)

            if mode != k:
                continue

            for transportation in self.agv_set[k].transportation_set:
                i = transportation[0]
                j = transportation[1]
                x_start = self.transportation_set[i][j].start_time.X
                y_start = i - 1 + half_space_width
                x_end = self.transportation_set[i][j].start_time.X + self.travel_time_mtx[i - 1][i]
                y_end = i - half_space_width
                dx = self.operation_set[i][j].start_time.X - x_end
                dy = 0
                plt.plot([x_start, x_end], [y_start, y_end], linewidth=1.5, color=self.agv_set[k].color)
                plt.plot([x_end, x_end + dx], [y_end, y_end + dy], linewidth=1.5, color=self.agv_set[k].color)

            for empty_trip in self.agv_set[k].empty_trip_set:
                i, j = empty_trip[0], empty_trip[1]
                i_, j_ = empty_trip[2], empty_trip[3]

                sgn = -1 if i >= i - 1 else 1

                x_3 = self.transportation_set[i_][j_].start_time.X
                y_3 = i_ - 1 - sgn * half_space_width
                x_2 = self.transportation_set[i_][j_].start_time.X - self.travel_time_mtx[i][i_ - 1]
                y_2 = i - half_space_width
                x_1 = self.operation_set[i][j].start_time.X
                y_1 = i - half_space_width

                plt.plot([x_1, x_2], [y_1, y_2], linewidth=1.5, linestyle='dashed', color=self.agv_set[k].color)
                plt.plot([x_2, x_3], [y_2, y_3], linewidth=1.5, linestyle='dashed', color=self.agv_set[k].color)


    def visualize_operation_m1(self):
        half_bar_width = 0.1
        i = 1
        for j in range(1, len(self.operation_set[i])):
            a = self.operation_set[i][j].start_time.X
            b = self.operation_set[i][j].start_time.X + self.operation_set[i][j].time
            plt.fill_betweenx(y=[i - half_bar_width, i + half_bar_width], \
                                x1=a, x2=b, color='gray', alpha=0.3, edgecolor='black')
            plt.text((a + b) / 2, i, f"Job {j}",ha="center", va="center", fontsize=10)


    def create_legend(self):
        legends = []
        legends.append(Patch(facecolor='gray', edgecolor='black', alpha=0.3, \
                             label="Operation without transportation"))
        for k in range(1, len(self.agv_set)):
            legends.append(Patch(facecolor=self.agv_set[k].color, edgecolor='black', alpha=0.3, \
                                 label=f"Operation transported by AGV {k}"))
            legends.append(Line2D([0], [0], color=self.agv_set[k].color, lw=1, \
                                  label=f'Transportation by AGV {k}'))
            legends.append(Line2D([0], [0], color=self.agv_set[k].color, lw=1, \
                                  label=f'Empty trip by AGV {k}', linestyle="--"))

        plt.legend(handles=legends, loc='lower right')

    
    def log_result(self):
        print(" ====================== RESULT ======================")
        print(f" - Objective value (makespan): {self.Cmax.X}")
        print(f" - Starting time of the all operations: ")

        for i in range(1, len(self.operation_set)):
            print(f" - - Machine {i}: ")
            for j in range(1, len(self.operation_set[i])):
                print(f" - - - Job {j}: {self.operation_set[i][j].start_time.X}")

        print(f" - AGV tasks and starting time of the all transportation: ")

        for i in range(2, len(self.transportation_set)):
            print(f" - - Machine {i}: ")
            for j in range(1, len(self.transportation_set[i])):
                print(f" - - - Job {j} by AGV {self.transportation_set[i][j].getAGV()}: \
                      {self.transportation_set[i][j].start_time.X}")
