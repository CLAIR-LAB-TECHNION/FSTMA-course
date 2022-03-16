import os
import time


def something_to_import():
    print(f'imported successfully from {os.path.relpath(__file__)}')
