FROM --platform=linux/amd64 rust:1.95-slim AS builder

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main(){}' > src/main.rs && \
    RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release && \
    rm -rf src

COPY spec/resources ./spec/resources
COPY src ./src
RUN touch src/main.rs && \
    RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release


FROM --platform=linux/amd64 gcr.io/distroless/cc-debian12 AS runtime

COPY --from=builder /build/target/release/rinha-2026 /app

ENTRYPOINT ["/app"]
