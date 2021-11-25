#  dct_mct_analysis
#  Copyright (C) 2021  FAU - RKI
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
import os
from pprint import pprint

import numpy as np
import pandas as pd

from i2mb.engine.agents import AgentList


class Experiment:
    def __init__(self, run_id, config):
        self.config = config
        self.population = AgentList(config["population_size"])
        self.run_id = int(run_id)
        self.data_dir = self.config.get("data_dir", "./")
        self.scenario_name = config.get("scenario", {}).get("name", "test")
        self.name = self.config.get("experiment_name", "dct_v_mct")
        self.config_name = self.config.get("name", "default")
        self.overwrite_files = self.config.get("overwrite_files", False)
        self.save_files = self.config.get("save_files", False)
        self.sim_engine = None

        # Structures that hold data updated every frame
        self.time_series_stats = []

        # Aggregated data at the end of the run
        self.agent_history = {}
        self.final_agent_stats = {}

    def frame_generator(self):
        for frame, _ in enumerate(self.sim_engine.engine.step()):
            # Stopping criteria
            stop = self.process_stop_criteria(frame)
            if stop:
                return

            yield frame

    def run_sim_engine(self):
        np.random.seed()
        if self.save_files and not self.overwrite_files and self.all_files_exist():
            print(f"Run {self.run_id} skipped. Files exist.")
            return

        self.display_start_msg()

        for frame in self.frame_generator():
            self.collect_time_series_data(frame)
            self.process_trigger_events(frame)

        self.collect_aggregated_data()

        if self.save_files:
            self.save_data()

        print(f"Run {self.get_base_name()} finished.")

    def process_stop_criteria(self, frame):
        raise RuntimeError("Experiment stop criteria needs to be implemented in child class.")

    def collect_time_series_data(self, frame):
        raise RuntimeError("collect_time_series_data needs to be implemented in child class.")

    def collect_aggregated_data(self):
        raise RuntimeError("collect_aggregated_data needs to be implemented in child class.")

    def process_trigger_events(self, frame):
        raise RuntimeError("process_trigger_events needs to be implemented in child class.")

    def display_start_msg(self):
        if self.config_name is None:
            print(f"Run {self.get_base_name()} Started.")
        else:
            print(f"Run {self.get_base_name()} Started with config {self.config_name}.")

    def create_directories(self):
        dir_ = self.get_experiment_dir()
        if not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)

        if not os.path.exists("/tmp/sct"):
            os.makedirs("/tmp/sct", exist_ok=True)

    def get_experiment_dir(self):
        return os.path.join(self.data_dir, self.config_name, self.name, self.scenario_name)

    def get_base_name(self):
        return f"{self.name}_{self.scenario_name}_i2bm_sim_data_{self.run_id:04}"

    def get_filename(self):
        base_name = self.get_base_name()
        dir_name = self.get_experiment_dir()
        return os.path.join(dir_name, base_name)

    def save_data(self):
        self.create_directories()
        self.write_meta_data()
        df = pd.DataFrame(self.final_agent_stats)
        key = f"end_values"
        df.to_hdf(f"{self.get_filename()}.hdf", key=key)

        df = pd.DataFrame(self.time_series_stats)
        df.to_hdf(f"{self.get_filename()}_results.hdf", key="results")

        np.savez(f"{self.get_filename()}.npz", **self.agent_history)
        # TODO: Output as csv (command line option)

    def file_exists(self, suffix):
        file_name = f"{self.get_filename()}{suffix}"
        file_exists = os.path.exists(file_name)
        if file_exists:
            print(f"Found File: {file_name}")

        return file_exists

    def all_files_exist(self):
        npz_file_exists = self.file_exists(".npz")
        hdf_file_exists = self.file_exists(".hdf")
        results_hdf_file_exists = self.file_exists("_results.hdf")
        if (npz_file_exists or hdf_file_exists or results_hdf_file_exists) and self.overwrite_files:
            print("Some files were found for this run and they will be overwritten.")

        return npz_file_exists and hdf_file_exists and results_hdf_file_exists

    def write_meta_data(self):
        meta_data_file_name = os.path.join(self.get_experiment_dir(), "meta_data")
        if os.path.exists(meta_data_file_name):
            return

        with open(meta_data_file_name, "wt") as out:
            pprint(self.config, stream=out)
