FROM python:3.11-slim

WORKDIR /app
COPY load_tests/requirements-load-test.txt /app/load_tests/requirements-load-test.txt
RUN pip install --no-cache-dir -r /app/load_tests/requirements-load-test.txt

COPY load_tests/locustfile.py /app/load_tests/locustfile.py
COPY backend/data/sample_test.csv /app/backend/data/sample_test.csv

EXPOSE 8089
CMD ["locust", "-f", "/app/load_tests/locustfile.py", "--web-host", "0.0.0.0", "--web-port", "8089"]
