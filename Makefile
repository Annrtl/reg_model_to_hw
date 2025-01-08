data:
	mkdir $@

data/house_prices/train.csv: data
	$(error 'Please download train.csv from https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data and install it in data/house_prices/')

data/house_prices/test.csv: data
	$(error 'Please download test.csv from https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data and install it in data/house_prices/')

database: data/house_prices/train.csv data/house_prices/test.csv

build:
	mkdir build

build/Makefile: build
	cmake -B build

setup: build/Makefile

model/model.json: model/model.py data/house_prices/train.csv
	python3 model/model.py

model: model/model.json