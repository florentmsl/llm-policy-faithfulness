dry:
	uv run python run.py --file experiments/blinded.yml --dry

experiments-dry:
	uv run python run.py --file experiments/blinded.yml --dry

experiments-run:
	uv run python run.py --file experiments/blinded.yml

aggregate:
	uv run python aggregate.py
