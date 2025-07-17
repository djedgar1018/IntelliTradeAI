"""
AI-Powered Trading Agent
Main entry point for the trading agent application
"""

import streamlit as st
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dashboard.streamlit_app import main as dashboard_main

def main():
    """Main function to run the AI Trading Agent"""
    # Run the dashboard
    dashboard_main()

if __name__ == "__main__":
    main()
