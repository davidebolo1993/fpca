# Dockerfile for fpca (C++ PCA tool)
FROM ubuntu:22.04

# Basic utilities
RUN apt-get update && \
    apt-get install -y git build-essential cmake wget pkg-config ca-certificates && \
    apt-get clean

# Install Armadillo and OpenBLAS via apt (fast, widely used)
RUN apt-get update && \
    apt-get install -y libopenblas-dev liblapack-dev libarmadillo-dev && \
    apt-get clean

# Optional: Show installed versions for debugging
RUN g++ --version && \
    armadillo-config --version || true

# Clone your repo
WORKDIR /opt
RUN git clone https://github.com/davidebolo1993/fpca.git
WORKDIR /opt/fpca

# Build fpca (assumes your repo contains fast_pca.cpp or CMakeLists.txt)
RUN mkdir -p build
WORKDIR /opt/fpca/build
RUN g++ -std=c++17 -O3 -march=native -fopenmp -I/usr/include -L/usr/lib -Wl,-rpath,/usr/lib -o fpca ../fpca.cpp -larmadillo -lopenblas -llapack

# Final container setup
WORKDIR /data
ENTRYPOINT ["/opt/fpca/build/fpca"]

# Usage example (with mounted /data volume):
# docker build -t fpca .
# docker run --rm -v $PWD//data fpca <input.tsv> <output.tsv> <n_pcs> <covariates.txt>

