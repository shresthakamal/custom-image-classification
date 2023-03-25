requirements:
	poetry export -f requirements.txt --output requirements.txt

install:
	pip3 install -r requirements.txt

tensorboard:
	tensorboard --logdir=lglutide/runs

app:
	python3 -m api.app

ngrok:
	ngrok http 5000
