import os
from collections import defaultdict
import csv

class EventWriter:
    def __init__(self, root):
        try:
            os.makedirs(root)
        except OSError:
            pass

        self.root = root
        self.events = dict()

    def add(self, key, event):
        f = None
        w = None

        if key not in self.events:
            f = open(f"{self.root}/{key}.csv", "w+")
            self.events[key] = f
            w = csv.DictWriter(f, event.keys(), delimiter=",")
            w.writeheader()
        else:
            f = self.events[key]
            w = csv.DictWriter(f, event.keys(), delimiter=",")

        w.writerow(event)
        f.flush()
