FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq
    'command -v ssh-agent >/dev/null || ( apt-get install -qq openssh-client )'
    apt-get install -qq curl
    curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain none
    apt-get install -qq build-essential clang cmake gfortran libopenblas-dev libssl-dev pkg-config
    rustup toolchain install stable --profile minimal
    rustup component add clippy rustfmt