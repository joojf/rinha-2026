FROM --platform=linux/amd64 rust:1.95-slim AS builder

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src/bin benches && \
    echo 'fn main(){}' > src/main.rs && \
    echo 'fn main(){}' > src/bin/build_index.rs && \
    touch src/lib.rs && \
    echo 'fn main(){}' > benches/score.rs && \
    RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release && \
    rm -rf src benches

COPY data ./data
COPY src ./src
RUN mkdir -p benches && echo 'fn main(){}' > benches/score.rs && \
    touch src/lib.rs src/main.rs && \
    RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release


FROM --platform=linux/amd64 gcr.io/distroless/cc-debian12 AS runtime

COPY --from=builder /build/target/release/rinha-2026 /app

ENTRYPOINT ["/app"]
