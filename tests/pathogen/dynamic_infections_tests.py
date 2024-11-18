import unittest
from multiprocessing import Pool

import numpy as np
import pandas as pd

from i2mb.engine.agents import AgentList
from i2mb.pathogen import SymptomLevels
from i2mb.pathogen.dynamic_infection import RegionVirusDynamicExposureBaseOnViralLoad
from i2mb.utils import global_time
from tests.pathogen import kissler_model


def exposure_function(t, region_exposed_contacts, region_contacts, distances, population_index,
                      infectiousness_level):
    dist_mat = ((region_contacts == population_index[:, None, None]).astype(float) *
                np.exp(-distances).reshape(1, -1, 1))
    dist_mat = dist_mat.sum(axis=2)
    mat = (region_contacts == population_index[:, None, None]).astype(float).sum(axis=2)
    inf_agents = infectiousness_level[population_index].ravel()
    inf_mat = mat * inf_agents.reshape(-1, 1)
    delta = inf_mat[(inf_agents > 0), :].dot((mat * dist_mat).T).sum(axis=0) * (inf_agents == 0)
    return delta


def recovery_function(t, time_exposed, exposure):
    t = time_exposed
    t = 20 / global_time.time_scalar * (t - 0.55 * global_time.make_time(day=1))

    cdf = 1 - 1 / (1 + np.exp(-t))
    return exposure * cdf


def get_symptom_levels_summary(population):
    return (population.symptom_level == SymptomLevels).sum(axis=0)


class SymptomLevelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.population_size = 1000
        self.population = AgentList(self.population_size)
        self.pathogen = RegionVirusDynamicExposureBaseOnViralLoad(
            population=self.population,
            exposure_function=exposure_function,
            recovery_function=recovery_function,
            infectiousness_function=kissler_model.triangular_viral_load,
            symptom_distribution=[0.4, 0.4, .138, .062],
            death_rate=[0.02, 0.05],
            icu_beds=self.population_size * 0.03,  # TODO move to scenario - Replace by available care
            # Function that changes the death rate based on care availability.
            # death_rate_function=lambda x: False,
            clearance_duration_distribution=kissler_model.clearance_period,
            proliferation_duration_distribution=kissler_model.proliferation_period,
            symptom_onset_estimator=kissler_model.compute_symptom_onset,
            max_viral_load_distribution=kissler_model.maximal_viral_load,
            max_viral_load=kissler_model.log_rna(0),
            min_viral_load=kissler_model.log_rna(40),
        )

        self.population.add_property("location", np.full(self.population_size, 1))
        self.population.add_property("at_home", np.full(self.population_size, True))

    def test_symptom_level_distribution(self):
        for t in range(self.population_size // 5):
            self.pathogen.introduce_pathogen(5, t, skip_incubation=False, asymptomatic=None)

        symptom_level_summary = get_symptom_levels_summary(self.population)
        population_affected = (self.population.state == [1, 2, 4, 5]).sum().sum()
        symptom_level_distribution = symptom_level_summary[1:] / population_affected
        error = np.abs(symptom_level_distribution[np.array(SymptomLevels.full_symptom_levels())] -
                       self.pathogen.symptom_distribution) <= 0.1
        self.assertListEqual([True] * len(SymptomLevels.full_symptom_levels()), list(error))

    def test_symptom_level_distribution_with_forced_percentage(self):
        for t in range(self.population_size // 5):
            self.pathogen.introduce_pathogen(5, t, skip_incubation=False, asymptomatic=0.4)

        symptom_level_summary = get_symptom_levels_summary(self.population)
        population_affected = (self.population.state == [1, 2, 4, 5]).sum().sum()
        symptom_level_distribution = symptom_level_summary[1:] / population_affected
        error = np.abs(symptom_level_distribution[np.array(SymptomLevels.full_symptom_levels())] -
                       self.pathogen.symptom_distribution) <= 0.1
        self.assertListEqual([True] * len(SymptomLevels.full_symptom_levels()), list(error))

    def test_profile_Creation_multiprocessing(self):
        tasks = []
        results = []
        with Pool(3, maxtasksperchild=1) as pool:
            for run in range(10):
                task = pool.apply_async(create_standalone_pathogen)
                tasks.append(task)

            for res in tasks:
                results.append(res.get())

        print(pd.concat([pd.DataFrame(r) for r in results], axis=1))


def create_standalone_pathogen():
    np.random.seed()
    population_size = 1000
    population = AgentList(population_size)
    pathogen = RegionVirusDynamicExposureBaseOnViralLoad(
        population=population,
        exposure_function=exposure_function,
        recovery_function=recovery_function,
        infectiousness_function=kissler_model.triangular_viral_load,
        symptom_distribution=[0.4, 0.4, .138, .062],
        death_rate=[0.02, 0.05],
        icu_beds=population_size * 0.03,  # TODO move to scenario - Replace by available care
        # Function that changes the death rate based on care availability.
        # death_rate_function=lambda x: False,
        clearance_duration_distribution=kissler_model.clearance_period,
        proliferation_duration_distribution=kissler_model.proliferation_period,
        symptom_onset_estimator=kissler_model.compute_symptom_onset,
        max_viral_load_distribution=kissler_model.maximal_viral_load,
        max_viral_load=kissler_model.log_rna(0),
        min_viral_load=kissler_model.log_rna(40),
    )

    population.add_property("location", np.full(population_size, 1))
    population.add_property("at_home", np.full(population_size, True))

    return population.incubation_duration

if __name__ == '__main__':
    unittest.main()
