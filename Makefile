requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

install:
	pip install -r requirements.txt

train:
	python -m lglutide.main

tensorboard:
	tensorboard --logdir=lglutide/runs

inference:
	python -m lglutide.predict

app:
	python -m api.app
