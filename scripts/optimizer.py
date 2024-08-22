import numpy as np
import components

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt

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

        


    def visualize_result(self):
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

        self.visualize_operation()
        self.visualize_transportation()

        # Set labels and title
        plt.xlabel("Time")
        plt.ylabel("Machine")
        plt.title("Manufacturing schedule chart")

        # Show the legend
        plt.legend()

        # Show the plot
        plt.show()


    def visualize_operation(self):
        half_bar_width = 0.1
        for i in range(1, len(self.operation_set)):
            for j in range(1, len(self.operation_set[i])):
                a = self.operation_set[i][j].start_time.X
                b = self.operation_set[i][j].start_time.X + self.operation_set[i][j].time
                plt.fill_betweenx(y=[i - half_bar_width, i + half_bar_width], \
                                  x1=a, x2=b, color='red', alpha=0.5, edgecolor='black')
                plt.text((a + b) / 2, i, f"Job {j}",ha="center", va="center", fontsize=10)

    
    def visualize_transportation(self):
        half_bar_width = 0.2
        # Get the data limits for x and y axes
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Calculate the scaling factor for the arrowhead
        scale_factor = (y_max - y_min) / (x_max - x_min)
        for i in range(2, len(self.transportation_set)):
            for j in range(1, len(self.transportation_set[i])):
                # Finding which AGV
                k = self.transportation_set[i][j].getAGV()

                x_start = self.transportation_set[i][j].start_time.X
                y_start = i - 1 + half_bar_width
                x_end = self.transportation_set[i][j].start_time.X + self.travel_time_mtx[i - 1][i]
                y_end = i - half_bar_width
                dx = self.operation_set[i][j].start_time.X - x_end
                dy = 0
                plt.plot([x_start, x_end], [y_start, y_end], linewidth=1.5, color='blue')
                plt.plot([x_end, x_end + dx], [y_end, y_end + dy], linewidth=1.5, color='blue')

                x_center = (x_start + x_end) / 2
                y_center = (y_start + y_end) / 2
                plt.text(x_center - 0.5, y_center + 0.1, f"{k}", ha="center", va="center")
