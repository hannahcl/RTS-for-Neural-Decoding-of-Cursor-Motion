# EKF-for-Neural-Decoding-of-Cursor-Motion
Project for the course Computational Neuro Science at KASIT


# Quick start
```bash
# initialize conda environment
conda create -n mvc-clip python=3.8
conda activate mvc-clip

# install requirements
pip install -r requirements.txt
pip install -e .

# Run pipeline
python scripts/generate_data
python scripts/fit_model
python scripts/run_filters

# Test only segments. replace test_ with the test you want to run
cd tests
python -m pytest -k test_generate_and_print_data -s
```
