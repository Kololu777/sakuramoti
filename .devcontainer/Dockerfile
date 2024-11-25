FROM nvcr.io/nvidia/pytorch:24.02-py3
ARG USERNAME=vscode

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=ja_JP.UTF-8 \
    TZ=Asia/Tokyo \
    apt_get_server=ftp.jaist.ac.jp/pub/Linux \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN \
    # apt
    --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    rm /etc/apt/apt.conf.d/docker-clean \
    && sed -i s@archive.ubuntu.com@${apt_get_server}@g /etc/apt/sources.list \
    && curl -sL https://raw.githubusercontent.com/eza-community/eza/main/deb.asc | gpg --dearmor -o /etc/apt/keyrings/gierens.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/gierens.gpg] http://deb.gierens.de stable main" | tee /etc/apt/sources.list.d/gierens.list \
    && chmod 644 /etc/apt/keyrings/gierens.gpg /etc/apt/sources.list.d/gierens.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    && apt-get install -y pipx
# FROM mcr.microsoft.com/vscode/devcontainers/python:3.12
# https://exerror.com/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directory/
RUN apt-get update && apt install -y libgl1-mesa-glx

RUN useradd -m vscode \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USERNAME}
