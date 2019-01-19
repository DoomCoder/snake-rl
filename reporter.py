from collections import deque
import numpy as np


class Reporter:
    def __init__(self, stats_n_episodes, report_freq, max_episodes):
        self.stats_n_episodes = stats_n_episodes
        self._memory = deque([], stats_n_episodes)
        self.report_freq = report_freq
        self.n_episodes = 0
        self.max_episodes = max_episodes

    def remember(self, n_steps, snake_len, reward, epsilon):
        self._memory.append((n_steps, snake_len, reward, epsilon))
        self.n_episodes += 1

    def get_report_str(self):
        arr = np.array(self._memory)
        means = np.mean(arr, axis=0)
        report_format = 'Ep {}/{} (eps {:.2f}) | avg steps: {}, avg len: {}, ' \
                        'avg reward: {} (last {} games)'
        report_data = (
            self.n_episodes, self.max_episodes, means[3], means[0], means[1],
            means[2], self.report_freq
        )

        return report_format.format(*report_data)

    def wants_to_report(self):
        return self.n_episodes % self.report_freq == 0
