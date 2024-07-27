.PHONY: install_rye install get_model format clean

SRC=quizscatterer
TESTS=tests

install_rye:
	curl -sSf https://rye.astral.sh/get | bash

install:
	rye sync
	mkdir -p ./.venv/lib/python3.12/site-packages/unidic/dicdir
	touch ./.venv/lib/python3.12/site-packages/unidic/dicdir/mecabrc
	
get_model:
	bash ./getGensimModel.sh

format:
	rye run ruff format $(SRC)
	rye run ruff check $(SRC) --fix-only
	rye run ruff format $(TESTS)
	rye run ruff check $(TESTS) --fix-only

clean:
	rm -rfv .venv
	rm -rfv $(SRC)/gensim_model/*