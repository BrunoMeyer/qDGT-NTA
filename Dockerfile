FROM python:3.10-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[notebook]"
COPY assets ./assets
COPY examples ./examples
COPY notebooks ./notebooks
COPY ["supplementary file.xlsx", "./supplementary file.xlsx"]
ENTRYPOINT ["qdgt-nta"]
CMD ["figures", "--output", "/app/figures"]
