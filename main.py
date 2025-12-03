"""
Main entry point for Stock Price Prediction System.
This script can be used as the main executable entry point.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from cli_interface import main

if __name__ == "__main__":
    main()
