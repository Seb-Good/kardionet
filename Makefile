build:
	docker-compose build

run:
	docker-compose run dev_env

notebook:
	docker-compose run --service-ports dev_env jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
