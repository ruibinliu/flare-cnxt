# Requirement
This project can only be run in Linux OS. Windows is not supported according to the Nvidia Flare.

# Project configuration
This project use .env to manage all the configurations. Use the following command to copy the example config, and customize according to your need.
```shell
cp .env.example .env
```

# Project setup
```shell
# Create virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

# Prepare data
Copy all the data to the runtime data root. Change the $RUNTIME_DATA_ROOT in the .env when needed.
```shell
python -m flare_app.prepare_data
```

# Start the training
```shell
python -m flare_app.job.py
```
