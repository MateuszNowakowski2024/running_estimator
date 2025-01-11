#!/bin/bash

# Example startup script
# 1. Fetch secrets from AWS Parameter Store
# 2. Start the Streamlit app

# Navigate to the script's directory (where app.py and fetch_secrets.py are located)
cd "$(dirname "$0")"

# Execute the Python script to fetch secrets and set environment variables
python3 fetch_secrets.py

# After secrets are loaded into environment variables, start your Streamlit app on port 8502
streamlit run app.py --server.port 8502
