import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class TimeBudgetedIterator(object):
    def __init__(self, iter_, budget):
        self.iter_ = iter_
        self.budget = budget
        self.time_elapsed = 0

    def __iter__(self):
        self.time_elapsed = 0
        count = 1
        for item in self.iter_:
            t0 = time.time()
            yield item
            tt = time.time()
            self.time_elapsed += tt - t0
            average_iteration_time = self.time_elapsed / count

            if self.budget > 0 and average_iteration_time * (count + 1) > self.budget:
                logger.warning("Going to exceed time limit, terminating loop.")
                break
            count += 1

    def __len__(self):
        return len(self.iter_)
