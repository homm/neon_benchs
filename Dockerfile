FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Update packages list. Disable cleanup to reuse pkg cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked set -ex \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt update \
	&& apt install --no-install-recommends -y \
		clang make \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /neon_benchs

ADD . /neon_benchs

RUN make arm
