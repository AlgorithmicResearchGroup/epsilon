FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    patchelf \
    wget \
    jq \
    ripgrep \
    procps \
    netcat-openbsd \
    iputils-ping \
    less \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m agent
WORKDIR /home/agent

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pyinstaller

COPY agent/ agent/
COPY agent_protocol/ agent_protocol/
COPY orchestrators/ orchestrators/
COPY runtime/ runtime/
COPY scripts/ scripts/
COPY orchestrate.py manifest.json challenge_prompts.json ./

RUN mkdir -p /home/agent/bin /tmp/pyi-build /tmp/pyi-dist /tmp/pyi-spec \
    && python -m PyInstaller \
      --noconfirm \
      --clean \
      --onefile \
      --collect-data litellm \
      --collect-data tiktoken \
      --collect-submodules tiktoken_ext \
      --hidden-import tiktoken_ext.openai_public \
      --name agent-main-linux \
      --workpath /tmp/pyi-build \
      --distpath /tmp/pyi-dist \
      --specpath /tmp/pyi-spec \
      runtime/agent_main.py \
    && python -m PyInstaller \
      --noconfirm \
      --clean \
      --onefile \
      --collect-data litellm \
      --collect-data tiktoken \
      --collect-submodules tiktoken_ext \
      --hidden-import tiktoken_ext.openai_public \
      --name byoa-runner-linux \
      --workpath /tmp/pyi-build \
      --distpath /tmp/pyi-dist \
      --specpath /tmp/pyi-spec \
      runtime/byoa_runner.py \
    && cp /tmp/pyi-dist/agent-main-linux /home/agent/bin/agent-main \
    && cp /tmp/pyi-dist/byoa-runner-linux /home/agent/bin/byoa-runner \
    && chmod +x /home/agent/bin/agent-main /home/agent/bin/byoa-runner \
    && rm -f /home/agent/runtime/agent_main.py /home/agent/runtime/byoa_runner.py \
    && rm -rf /tmp/pyi-build /tmp/pyi-dist /tmp/pyi-spec

USER agent
ENV PYTHONUNBUFFERED=1
ENV COLLAB_AGENT_MAIN_BINARY=/home/agent/bin/agent-main
ENV COLLAB_BYOA_RUNNER_BINARY=/home/agent/bin/byoa-runner
ENV COLLAB_DOCKER_AGENT_MAIN_BINARY=/home/agent/bin/agent-main
ENV COLLAB_DOCKER_BYOA_RUNNER_BINARY=/home/agent/bin/byoa-runner

ENTRYPOINT ["python", "orchestrate.py", "--pattern", "dag"]
