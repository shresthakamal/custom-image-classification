requirements:
	poetry export -f requirements.txt --output requirements.txt

install:
	pip install -r requirements.txt

tensorboard:
	tensorboard --logdir=lglutide/runs

app:
	python3 -m api.app

ngrok:
	ngrok http 5000
