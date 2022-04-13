import numpy as np

from i2mb import Model
from i2mb.pathogen import UserStates, SymptomLevels


class GetTested(Model):
    def __init__(self, population, delay=2, test_to_leave=False, test_household=False,
                 delay_test=None,
                 quarantine_duration=5, share_test_result=1.):
        self.share_test_result = share_test_result
        self.quarantine = quarantine_duration
        self.test_household = test_household
        self.test_to_leave = test_to_leave
        self.delay = delay
        self.population = population

        self.code = -1

        self.test_result_current = np.zeros((len(population), 1), dtype=bool)
        self.time_of_last_test = np.zeros((len(population), 1), dtype=bool) - 1

        self.self_isolated = 0

        def default_number():
            return 5

        self.delay_test = default_number

        if isinstance(delay_test, int):
            def function_number():
                return delay_test

            self.delay_test = function_number

        elif delay_test is not None:
            self.delay_test = delay_test

    def post_init(self, base_file_name=None):
        super(GetTested, self).post_init(base_file_name)
        if hasattr(self.population, "register"):
            self.code = self.population.register("Self isolation")

    def step(self, t):
        # reset counter
        self.self_isolated = 0
        self.collect_results(t)
        self.update_time_of_last_test_for_quarantined_agents()
        # self.invalidate_test_results()
        self.request_test_for_symptomatic_agents(t)
        self.request_agents_with_positive_test_to_isolate()

        if not self.population.isolated.any():
            return

        self.determine_who_can_leave_isolation(t)

    def determine_who_can_leave_isolation(self, t):
        # Determine who can leave isolation
        leave = self.population.state == UserStates.immune
        # Add asymptomatic agents this brings in susceptible and exposed agents
        leave |= (self.population.state == UserStates.exposed)
        leave |= (self.population.state == UserStates.susceptible)
        leave |= (self.population.symptom_level == SymptomLevels.no_symptoms)
        leave &= self.population.isolated
        if self.test_to_leave:
            # update test information
            if self.test_household:
                # Ensure no one in the household is infected
                affected = ~leave & self.population.isolated
                lock_down = (self.population.home.reshape((-1, 1)) ==
                             self.population.home[affected.ravel()]).any(axis=1)
                leave &= ~lock_down.reshape((-1, 1))

            negative_test_results = ~self.population.test_result & self.test_result_current
            self.test_result_current[leave] = False

            get_tests = leave & ((t - self.time_of_last_test) >= self.delay_test()) & ~negative_test_results
            self.request_test(get_tests, "Test to leave")

            leave &= negative_test_results

        else:
            leave &= (t - self.population.isolation_time) > self.quarantine

        if leave.any():
            self.population.leave_request[leave.ravel()] = True

    def request_agents_with_positive_test_to_isolate(self):
        # Request isolation of agents that tested positive
        candidates = self.population.test_result & self.test_result_current & ~self.population.isolated
        if candidates.any():
            self.self_isolated = candidates.sum()
            self.test_result_current[candidates] = False
            self.population.isolation_request[candidates.ravel()] = True
            self.population.isolated_by[candidates.ravel()] = self.code

    def invalidate_test_results(self):
        # Test results become invalid once agent becomes immune.
        deceased = (self.population.state == UserStates.deceased)
        recovered = (((self.population.state == UserStates.immune) | deceased) & self.test_result_current &
                     self.population.test_result)
        if recovered.any():
            self.test_result_current[recovered.ravel()] = False

    def request_test_for_symptomatic_agents(self, t):
        active = self.population.state == UserStates.infectious
        active &= self.population.symptom_level != SymptomLevels.no_symptoms
        candidates = (t - (self.population.time_of_infection + self.population.incubation_duration)) >= self.delay
        candidates &= active
        self.request_test(candidates, "Symptomatic agent")

    def request_test(self, petitioners, provenance):
        petitioners &= ~self.population.test_in_process & ~self.population.test_request & ~self.test_result_current
        if petitioners.any():
            petitioners = petitioners.ravel()
            self.population.test_request[petitioners] = True

    def collect_results(self, t):
        test_results = self.population.results_available.ravel()  # this is not a copy
        if test_results.any():
            self.test_result_current[test_results] = True
            self.time_of_last_test[test_results] = t

            # Check if we have to, and want to, report the positive test to the health authorities
            share_results = np.random.random((len(self.population), 1)) <= self.share_test_result
            report_tests = test_results & ~self.population.isolated.ravel() & share_results.ravel()
            if hasattr(self.population, "positive_test_report"):
                self.population.positive_test_report[report_tests] = True

            if hasattr(self.population, "dct_positive_test_report"):
                self.population.dct_positive_test_report[report_tests] = True

            if hasattr(self.population, "fnf_positive_test_report"):
                self.population.fnf_positive_test_report[report_tests] = True

            self.population.results_available[test_results] = False  # modifying this modifies test results

    def update_time_of_last_test_for_quarantined_agents(self):
        quarantined = self.population.isolated.ravel() & (self.time_of_last_test == -1).ravel()
        if quarantined.any():
            self.time_of_last_test[quarantined] = self.population.isolation_time[quarantined]
