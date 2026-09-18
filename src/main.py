import os
import sys
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Config imports
from config import logging_config

# Algorithm Imports
from algorithms.cb_adaptations import cb_nsga2, cb_nsga3, cb_sms_emoa, cb_spea2
from algorithms.cbd_adaptations import cbd_nsga2, cbd_nsga3
from algorithms.pd_adaptations import pd_nsga2, pd_nsga3

def main():
    # Set up logger
    logging_config.setup_logging()
    
    # Create algo instance
    inst = cb_nsga2.CB_NSGA2()

if __name__ == "__main__":
    main()