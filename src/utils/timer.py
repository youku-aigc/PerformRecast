# coding: utf-8
"""Tiny stopwatch helper."""
import time


class Timer(object):
    def __init__(self):
        self.start_time = 0.
        self.diff = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        return self.diff
