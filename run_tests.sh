#!/usr/bin/bash
python -m unittest tests.tests_base_trainer
python -m unittest tests.tests_data
python -m unittest tests.tests_metrics
python -m unittest tests.tests_trainers
python -m unittest tests.tests_callbacks
