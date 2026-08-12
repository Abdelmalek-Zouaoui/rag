FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "rag.cli"]
CMD ["--help"]
