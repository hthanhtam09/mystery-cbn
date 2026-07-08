# Pinned benchmark environment (BENCHMARK_SPEC.md §9.1).
# Absolute times are comparable only across runs of the same image digest;
# CI records the resolved digest in every report's machine fingerprint.
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /engine
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY benchmarks ./benchmarks
COPY assets ./assets

RUN pip install --no-cache-dir -e ".[dev,pdf]"

ENTRYPOINT ["pytest", "benchmarks", "-q"]
