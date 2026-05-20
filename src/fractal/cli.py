from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fractal")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="workspace directory to edit; defaults to the current directory",
    )
    parser.add_argument(
        "--lm", default="openai/gpt-5.5", help="DSPy LM model string for PredictRLM"
    )
    parser.add_argument(
        "--sub-lm", default="openai/gpt-5.1", help="DSPy sub-LM model string"
    )
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument(
        "--quiet", action="store_true", help="reserved for quieter terminal output"
    )
    parser.add_argument(
        "--debug", action="store_true", help="enable PredictRLM debug mode"
    )
    return parser


def run_tui(args: argparse.Namespace) -> int:
    from .runtime import FractalRuntime
    from .tui import TerminalFractalApp

    workspace = args.workspace.resolve()
    runtime = FractalRuntime.create(
        workspace_path=workspace,
        lm=args.lm,
        sub_lm=args.sub_lm,
        max_iterations=args.max_iterations,
        verbose=False,
        debug=args.debug,
    )
    asyncio.run(TerminalFractalApp(runtime).run())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_tui(args)


if __name__ == "__main__":
    raise SystemExit(main())
