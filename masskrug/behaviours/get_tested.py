import numpy as np

from masskrug import Model
from masskrug.pathogen import UserStates, SymptomLevels


class GetTested(Model):
    def __init__(self, population, delay=2, test_to_leave=False, test_household=False,
                 delay_test=5,
                 quarantine_duration=5):
        self.delay_test = delay_test
        self.quarantine = quarantine_duration
        self.test_household = test_household
        self.test_to_leave = test_to_leave
        self.delay = delay
        self.population = population

        self.code = -1

        self.test_result_current = np.zeros((len(population), 1), dtype=bool)

        self.self_isolated = 0

    def post_init(self):
        if hasattr(self.population, "register"):
            self.code = self.population.register("Self isolation")

    def step(self, t):
        # reset counter
        self.self_isolated = 0

        # Collect results
        test_results = self.population.results_available.ravel()  # this is not a copy
        if test_results.any():
            self.test_result_current[test_results] = True

            # Check if we have to report the positive test to the health authorities
            report_tests = test_results & ~self.population.isolated.ravel()
            if hasattr(self.population, "positive_test_report"):
                self.population.positive_test_report[report_tests] = True

            if hasattr(self.population, "dct_positive_test_report"):
                self.population.dct_positive_test_report[report_tests] = True

            if hasattr(self.population, "fnf_positive_test_report"):
                self.population.fnf_positive_test_report[report_tests] = True

            self.population.results_available[test_results] = False  # modifying this modifies testresults

        # Test results become invalid once agent becomes immune.
        alive = (self.population.state != UserStates.deceased)
        recovered = (((self.population.state == UserStates.immune) | ~alive) & self.test_result_current &
                     self.population.test_result)
        if recovered.any():
            self.test_result_current[recovered.ravel()] = False

        active = self.population.state == UserStates.infectious
        candidates = (t - (self.population.time_of_infection + self.population.incubation_period)) >= self.delay
        candidates &= active & ~self.population.test_in_process & ~self.population.test_result
        if candidates.any():
            candidates = candidates.ravel()
            self.population.test_request[candidates] = True
            self.test_result_current[candidates] = False

        # Request isolation of agents that tested positive
        candidates = self.population.test_result & self.test_result_current & ~self.population.isolated
        if candidates.any():
            self.self_isolated = candidates.sum()
            self.population.isolation_request[candidates.ravel()] = True
            self.population.isolated_by[candidates.ravel()] = self.code

        if not self.population.isolated.any():
            return

        # Determine who can leave isolation
        leave = self.population.state == UserStates.immune
        leave |= ((self.population.state == UserStates.exposed) |
                  (self.population.state == UserStates.susceptible))

        if self.test_to_leave:
            # update test information
            if self.test_household:
                # Ensure no one in the household is infected
                affected = ~leave & self.population.isolated
                lock_down = (self.population.home.reshape((-1, 1)) == self.population.home[affected.ravel()]).any(
                    axis=1)
                leave &= ~lock_down.reshape((-1, 1))

            get_tests = leave & self.population.isolated & ((t - self.population.isolation_time) >= self.delay_test)
            get_tests &= ~self.test_result_current & ~self.population.test_in_process
            self.population.test_request[get_tests.ravel()] = True
            self.test_result_current[get_tests] = False
            leave &= ~self.population.test_result & self.test_result_current
            leave[get_tests] = False
        else:
            # Add asymptomatic agents this brings in susceptible and exposed agents
            leave |= (self.population.symptom_level == SymptomLevels.no_symptoms)
            leave &= (t - self.population.isolation_time) > self.quarantine

        leave &= self.population.isolated
        if leave.any():
            self.population.leave_request[leave.ravel()] = True

            # Test results become invalid once agent leave isolation.
            self.test_result_current[leave.ravel()] = False
