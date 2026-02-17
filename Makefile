dry:
	uv run python run.py --file experiments/freeway.yml --dry

experiments-dry:
	uv run python run.py --file experiments/freeway.yml --dry

experiments-run:
	uv run python run.py --file experiments/freeway.yml

experiments-dry-pong:
	uv run python run.py --file experiments/pong.yml --dry

experiments-run-pong:
	uv run python run.py --file experiments/pong.yml
