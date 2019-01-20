from collections import deque
import numpy as np
import logging


class Reporter:
    def __init__(self, stats_n_episodes, max_episodes, log_to_file=True):
        self.stats_n_episodes = stats_n_episodes
        self._memory = deque([], stats_n_episodes)
        self.n_episodes = 0
        self.max_episodes = max_episodes
        if log_to_file:
            self.reportLogger=self._init_logger()
        else:
            self.reportLogger=None
        
    def _init_logger(self):
        reportLogger=logging.getLogger("report")
        reportLogger.setLevel(logging.DEBUG)
        handler = logging.FileHandler("report.log")
        handler.setLevel(logging.DEBUG)
        reportLogger.addHandler(handler)
        
        return reportLogger
    
    def remember(self, n_steps, snake_len, reward, epsilon):
        self._memory.append((n_steps, snake_len, reward, epsilon))
        self.n_episodes += 1

    def get_report_str(self):
        arr = np.array(self._memory)
        means = np.mean(arr, axis=0)
        report_format = 'Ep {:5d}/{} | eps {:3.2f} | steps: {:5.2f} | len: {:5.2f} | ' \
                        'reward: {:4.2f} | (avgs last {} games)'
        report_data = (
            self.n_episodes, self.max_episodes, means[3], means[0], means[1],
            means[2], self.stats_n_episodes
        )
        formated_report=report_format.format(*report_data)
        if self.reportLogger != None:
            self.reportLogger.info(formated_report)
        
        return formated_report

    def wants_to_report(self):
        return self.n_episodes % self.stats_n_episodes == 0
