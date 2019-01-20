import logging

import reporter
import numpy as np

class BaseLogger():
    def __init__(self, batch_size, autolog=True):
        self.batch_size = batch_size
        self.autolog = autolog
        self.memory = []

    def log(self, **kwargs):
        self.memory.append(kwargs)
        if self.autolog:
            if (len(self.memory) >= self.batch_size):
                self.report()
                self.memory = []

    def report(self):
        raise NotImplementedError


class AvgPerformenceReporter(BaseLogger):
    def report(self):
        keys = self.memory[0].keys()
        avg = self._aggregate_avg_memory()
        report=""
        for key in keys:
            report+=f"{key}: {avg[key]} | "

        print(report)

    def _aggregate_avg_memory(self):
        memory = self.memory
        keys = memory[0].keys()
        sums = {key: 0.0 for key in keys}
        for log in memory:
            for key in log:
                sums[key] += log[key]

        avg = {key: sums[key]/len(log) for key in sums}

        return avg

lossLogger=logging.getLogger("loss")
lossLogger.setLevel(logging.DEBUG)
handler = logging.FileHandler("loss.log")
handler.setLevel(logging.DEBUG)
lossLogger.addHandler(handler)


qLogger=logging.getLogger("avg_q_values")
qLogger.setLevel(logging.DEBUG)
handler = logging.FileHandler("avg_q.log")
handler.setLevel(logging.DEBUG)
qLogger.addHandler(handler)



class QLogger():
    def log(self, q_table: np.array):
        avg_q=np.mean(q_table)
        qLogger.info(avg_q)

def loss_logger_decorator(func):
    def wrapper(*args, **kwargs):
        loss = func(*args, **kwargs)
        lossLogger.info(loss)

    return wrapper
  