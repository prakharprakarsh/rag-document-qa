.PHONY: install run-api run-frontend run docker-build docker-up docker-down test clean

install:
	pip install -r requirements.txt

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	streamlit run frontend/app.py

run: ## Run both API and frontend (2 terminal tabs needed)
	@echo "Run these in separate terminals:"
	@echo "  make run-api"
	@echo "  make run-frontend"

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

test:
	python -m pytest tests/ -v

clean:
	rm -rf data/chroma_db
	find . -type d -name __pycache__ -exec rm -rf {} +