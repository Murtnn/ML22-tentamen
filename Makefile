
.DEFAULT: run

run:	
	export PYTHONPATH="$$HOME/code/tentamen" ;poetry run python dev/scripts/01_model_design.py

runGRU:	
	export PYTHONPATH="$$HOME/code/tentamen" ;poetry run python dev/scripts/01_model_GRU_design.py

runBEST:	
	export PYTHONPATH="$$HOME/code/tentamen" ;poetry run python dev/scripts/2c_model_design.py

tune: 
	export PYTHONPATH="$$HOME/code/tentamen" ;poetry run python dev/scripts/02_tune.py

tuneGRU: 
	export PYTHONPATH="$$HOME/code/tentamen" ;poetry run python dev/scripts/02_tune_GRU.py

result: 
	export PYTHONPATH="$$HOME/code/tentamen" ;poetry run python dev/scripts/03_result.py

format:
	poetry run isort dev
	poetry run black dev
	poetry run isort tentamen
	poetry run black tentamen

lint:
	poetry run flake8 dev
	poetry run flake8 tentamen
	poetry run mypy --no-strict-optional --warn-unreachable --show-error-codes --ignore-missing-imports dev
	poetry run mypy --no-strict-optional --warn-unreachable --show-error-codes --ignore-missing-imports tentamen


clean:
	rm -rf logs/*
	rm -rf models/*


