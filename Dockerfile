FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
COPY README.md LICENSE CITATION.cff CHANGELOG.md ./
COPY docs/ docs/
COPY src/ src/
RUN python -m pip install --no-cache-dir build \
    && python -m build --wheel

FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="MobiOrigin"
LABEL org.opencontainers.image.description="Sequence-and-marker classification of bacterial replicons"
LABEL org.opencontainers.image.source="https://github.com/Raza-pl/MobiOrigin"

ARG DIAMOND_VERSION=2.1.9
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 wget \
    && wget -q "https://github.com/bbuchfink/diamond/releases/download/v${DIAMOND_VERSION}/diamond-linux64.tar.gz" -O /tmp/diamond.tar.gz \
    && tar -xzf /tmp/diamond.tar.gz -C /usr/local/bin diamond \
    && rm /tmp/diamond.tar.gz \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/dist/*.whl /tmp/mobiorigin.whl
RUN python -m pip install --no-cache-dir /tmp/mobiorigin.whl \
    && rm /tmp/mobiorigin.whl

WORKDIR /work
ENTRYPOINT ["mobiorigin"]
CMD ["--help"]
