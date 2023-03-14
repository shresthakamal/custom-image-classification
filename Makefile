requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

install:
	pip install -r requirements.txt

train:
	python -m lglutide.main CUDA_VISIBLE_DEVICES=7
