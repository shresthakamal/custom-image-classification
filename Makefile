requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

install:
	pip install -r requirements.txt
