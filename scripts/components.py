from gurobipy import GRB

class Operation:
    def __init__(self, operation_time, model, machine_id, job_id) -> None:
        self.start_time = model.addVar(lb=0.0, ub=float("inf"), vtype=GRB.CONTINUOUS, \
                                       name=f"so^{job_id}/{machine_id}")
        self.time = operation_time


class Transportation:
    def __init__(self, num_agv, transportation_time, model, machine_id, job_id) -> None:
        """
            Each transportation has a list agv_state
                agv_state[i] == 1 if this transportation is carried out by AGV i
                agv_state[i] == 0 otherwise
        """
        self.agv_state = [0]
        self.start_time = model.addVar(lb=0.0, ub=float("inf"), vtype=GRB.CONTINUOUS, \
                                       name=f"st^{job_id}/{machine_id}")
        self.time = transportation_time
        for i in range(num_agv):
            self.agv_state.append(model.addVar(lb=0, ub=1, vtype=GRB.BINARY, \
                                  name=f"{i + 1}^x^{job_id}/{machine_id}"))
            
    def getAGV(self):
        for i in range(1, len(self.agv_state)):
            if self.agv_state[i].X == 1:
                return i
            
        return -1
    

class AGV:
    def __init__(self, color) -> None:
        self.transportation_set = []
        self.empty_trip_set = []
        self.color = color

    def add_transportation(self, machine, job, start_time):
        self.transportation_set.append([machine, job, start_time])

    def schedule_empty_trip(self):
        self.transportation_set.sort(key= lambda trans: trans[2])
        for i in range(len(self.transportation_set) - 1):

            new_empty_trip = [self.transportation_set[i][0], self.transportation_set[i][1], \
                              self.transportation_set[i + 1][0], self.transportation_set[i + 1][1]]
            self.empty_trip_set.append(new_empty_trip)
        
        pass
