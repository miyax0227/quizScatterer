.PHONY: install_rye install get_model format clean

SRC=quizscatterer

install_rye:
	curl -sSf https://rye.astral.sh/get | bash

install:
	rye sync

get_model:
	bash ./getGensimModel.sh

format:
	rye run ruff format $(SRC)
	rye run ruff check $(SRC) --fix-only

clean:
	rm -rfv .venv
	rm -rfv $(SRC)/gensim_model/*