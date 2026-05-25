"""Small command-line helpers for local vLLM experiments."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys


def _check(_: argparse.Namespace) -> int:
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        print("vLLM is not installed. Run: uv sync", file=sys.stderr)
        return 1

    print(f"vLLM installed: {version}")
    print(f"Python executable: {sys.executable}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    if args.dtype:
        command.extend(["--dtype", args.dtype])

    env = os.environ.copy()
    print("Starting vLLM OpenAI-compatible server:")
    print(" ".join(command))
    return subprocess.call(command, env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-exp",
        description="Helpers for a uv-managed vLLM experiment project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Verify that vLLM is installed.")
    check.set_defaults(func=_check)

    serve = subparsers.add_parser(
        "serve",
        help="Start the vLLM OpenAI-compatible API server.",
    )
    serve.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id or local model path.",
    )
    serve.add_argument("--host", default="0.0.0.0", help="Server bind host.")
    serve.add_argument("--port", type=int, default=8000, help="Server bind port.")
    serve.add_argument(
        "--dtype",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="Optional vLLM dtype override.",
    )
    serve.set_defaults(func=_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
