import numpy as np

from masskrug.interventions.base_intervention import Intervention
from masskrug.pathogen import UserStates
from masskrug.utils import global_time


def default_test(population):
    return (population.state == UserStates.infected) | (population.state == UserStates.infectious)


class Test(Intervention):
    def __init__(self, population, duration=1, test_method=None,
                 closing_time=None, opening_time=None):
        """
        This model represents the testing facilities of the simulated world. For an agent i to be tested, a test request
        is made by setting  'population.test_request[i] = True'. Results are given after duration has elapsed.

        :param population:
        :param duration: Wait time for test results to be available
        :param test_method: Callable function that takes as parameter a population and determines the value of the
                            test result. Defaults to calling :func:`default_test` which uses population.state to
                            determine if the patient should test positive.

        """

        if closing_time is None:
            closing_time = global_time.make_time(hour=16, minutes=30)

        if opening_time is None:
            opening_time = global_time.make_time(hour=8)

        self.closing_time = closing_time
        self.opening_time = opening_time

        if test_method is None:
            test_method = default_test

        self.test_method = test_method
        self.duration = duration
        self.population = population

        n = len(population)
        # Number of times a test was requested
        self.tested = np.zeros((n, 1), dtype=int)

        self.test_request = np.zeros((n, 1), dtype=bool)
        self.test_result = np.zeros((n, 1), dtype=bool)

        # Latest test date
        self.test_date = np.zeros((n, 1), dtype=int)
        self.results_available = np.zeros((n, 1), dtype=bool)
        self.test_in_process = np.zeros((n, 1), dtype=bool)

        population.add_property("tested", self.tested)
        population.add_property("test_request", self.test_request)
        population.add_property("test_result", self.test_result)
        population.add_property("test_date", self.test_date)
        population.add_property("results_available", self.results_available)
        population.add_property("test_in_process", self.test_in_process)

    def step(self, t):

        # Enforce opening hours
        time = global_time.time(t)
        if not (self.opening_time <= time < self.closing_time):
            return

        test_actors = self.population.test_request.ravel()
        if test_actors.any():
            current_closing_time = global_time.to_current(self.closing_time, t)
            self.population.test_date[test_actors] = np.random.randint(t, current_closing_time, test_actors.sum(
                                                                       )).reshape((-1, 1))
            self.population.results_available[test_actors] = False
            self.test_in_process[test_actors] = True
            self.population.test_request[test_actors] = False

        # Update result availability
        if self.test_in_process.any():
            results_available = (self.test_in_process & ((t - self.population.test_date) >
                                                         self.duration)).ravel()
            if results_available.any():
                self.population.test_request[results_available] = False
                self.population.tested[results_available] += 1
                self.population.results_available[results_available] = True
                self.population.test_result[results_available] = self.test_method(self.population[results_available])
                self.test_in_process[results_available] = False
