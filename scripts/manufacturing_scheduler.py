import numpy as np
import yaml
import csv
from pathlib import Path

import optimizer

INPUT_FILE = Path(__file__).parent / "../input/input.yaml"
MACHINE_PROCESS_TIME = Path(__file__).parent / "../input/machine_process_time.csv"
MACHINE_LAYOUT = Path(__file__).parent / "../input/machine_layout.csv"

class Scheduler:
    def __init__(self) -> None:
        self.input_set = {
            'input_data': {},
            'machine_process_time': [],
            'machine_layout_mtx': []
        }

        # Read the input file
        with open(INPUT_FILE, 'r') as file:
            self.input_set['input_data'] = yaml.safe_load(file)

        # Read machine configuration
        with open(MACHINE_PROCESS_TIME, "r") as file:
            reader = csv.reader(file, delimiter=",")
            result = np.array(list(reader)).astype("float")
            self.input_set['machine_process_time'] = result[0]

        with open(MACHINE_LAYOUT, "r") as file:
            reader = csv.reader(file, delimiter=",")
            result = np.array(list(reader)).astype("float")
            self.input_set['machine_layout_mtx'] = result

        assert len(self.input_set['machine_process_time']) == self.input_set['input_data']['num_machine'],\
            ("INPUT ERROR: the number of machine and the process time data is not match")
        
        assert len(self.input_set['machine_layout_mtx']) == self.input_set['input_data']['num_machine'],\
            ("INPUT ERROR: the number of machine and the layout data is not match")
        
        print("[Scheduler] Init: Get input successfully !!!")
        print(f"Number of job: {self.input_set['input_data']['num_job']}")
        print(f"Number of AGV: {self.input_set['input_data']['num_AGV']}")
        print(f"Number of machine: {self.input_set['input_data']['num_machine']}")
        print(f"Machine processing time: {self.input_set['machine_process_time']}")
        print(f"Distance between machines:")
        print(self.input_set['machine_layout_mtx'])

        self.optimize_engine = optimizer.MILPOptimizer(self.input_set)

    def execute(self):
        self.optimize_engine.solve()
        self.optimize_engine.visualize_result()
            

if __name__ == "__main__":
    s = Scheduler()
    s.execute()
