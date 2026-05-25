# vLLM uv Experiment

Minimal Python project for running vLLM with `uv`.

## Install uv

```bash
./scripts/install_uv.sh
```

Restart your shell if `uv` is not found immediately after install.

## Set up the project

```bash
uv sync
```

## Check the environment

```bash
uv run vllm-exp check
```

## Start an OpenAI-compatible vLLM server

```bash
uv run vllm-exp serve --model Qwen/Qwen2.5-0.5B-Instruct
```

Then call the server at `http://localhost:8000/v1`.

Notes:

- vLLM is primarily intended for Linux hosts with NVIDIA GPUs.
- On macOS, dependency installation or runtime support may be limited.
- Use a smaller model first to verify the setup before trying large models.
