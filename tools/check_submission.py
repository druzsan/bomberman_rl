"""Pre-flight gate for a submitted agent directory.

Everything here tests the *shipped copy*, not the working tree: the directory is
copied to a throwaway agent name and every check runs against that copy, so a
file that only exists in development shows up as a failure rather than as a
surprise on the graders' machine.

    uv run python -m tools.check_submission --agent q_tabular_agent
    uv run python -m tools.check_submission --agent q_tabular_agent --rounds 300 --zip
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AGENT_CODE = REPO / "agent_code"

def _required_arities() -> tuple[dict[str, int], dict[str, int]]:
    """Read the required callback arities from the engine itself.

    ``AgentRunner.__init__`` validates by argument *count* and raises at load
    time, so duplicating the numbers here would only create a second thing to
    keep in sync -- and did, once.
    """
    from agents import AGENT_API

    return ({k: len(v) for k, v in AGENT_API["callbacks"].items()},
            {k: len(v) for k, v in AGENT_API["train"].items()})


REQUIRED_CALLBACKS, REQUIRED_TRAIN = _required_arities()

FORBIDDEN_IMPORTS = ("multiprocessing", "torch.multiprocessing", "training")
FORBIDDEN_PATTERNS = [
    (re.compile(r"""["']/home/"""), "absolute POSIX path"),
    (re.compile(r"""["'][A-Za-z]:\\\\"""), "absolute Windows path"),
    (re.compile(r"\bfrom\s+training\b|\bimport\s+training\b"), "imports the training harness"),
]

MAX_MODEL_MB = 20.0
MAX_MEAN_MS = 10.0
MAX_P99_MS = 50.0


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""


class Checker:
    def __init__(self, agent: str, rounds: int):
        self.agent = agent
        self.rounds = rounds
        self.results: list[Result] = []
        self.ship_name = f"{agent}__shipcheck"
        self.ship_dir = AGENT_CODE / self.ship_name

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append(Result(name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        return ok

    # -- setup ------------------------------------------------------------
    def stage(self) -> None:
        src = AGENT_CODE / self.agent
        if self.ship_dir.exists():
            shutil.rmtree(self.ship_dir)
        shutil.copytree(src, self.ship_dir,
                        ignore=shutil.ignore_patterns("__pycache__", "logs", "*.log",
                                                      "USES_LIB"))
        # The engine imports by directory name, so the copy must be importable.
        (self.ship_dir / "__init__.py").touch()

    def cleanup(self) -> None:
        shutil.rmtree(self.ship_dir, ignore_errors=True)

    # -- static checks ----------------------------------------------------
    def check_files(self) -> None:
        for required in ("callbacks.py", "__init__.py"):
            self.record(f"{required} present", (self.ship_dir / required).exists())
        models = list(self.ship_dir.glob("*.npz")) + list(self.ship_dir.glob("*.pt"))
        if not self.record("model artifact present", bool(models),
                           ", ".join(m.name for m in models)):
            return
        size_mb = max(m.stat().st_size for m in models) / 1e6
        self.record(f"model under {MAX_MODEL_MB:.0f} MB", size_mb <= MAX_MODEL_MB,
                    f"{size_mb:.2f} MB")

    def check_source(self) -> None:
        bad = []
        for path in sorted(self.ship_dir.rglob("*.py")):
            text = path.read_text()
            rel = path.relative_to(self.ship_dir)
            for pattern, why in FORBIDDEN_PATTERNS:
                for line_no, line in enumerate(text.splitlines(), 1):
                    if line.lstrip().startswith("#"):
                        continue
                    if pattern.search(line):
                        bad.append(f"{rel}:{line_no} {why}")
            try:
                tree = ast.parse(text)
            except SyntaxError as exc:
                bad.append(f"{rel}: syntax error {exc}")
                continue
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                for name in names:
                    if any(name == f or name.startswith(f + ".") for f in FORBIDDEN_IMPORTS):
                        bad.append(f"{rel}:{node.lineno} imports {name}")
        self.record("no forbidden imports or absolute paths", not bad, "; ".join(bad[:5]))

    def check_relative_paths(self) -> None:
        """Every file the agent opens must be opened relatively (cwd is the agent dir)."""
        bad = []
        for path in sorted(self.ship_dir.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name not in ("open", "load", "savez", "savez_compressed"):
                    continue
                for arg in node.args[:1]:
                    if (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                            and (arg.value.startswith("/")
                                 or re.match(r"^[A-Za-z]:", arg.value))):
                        bad.append(f"{path.name}:{node.lineno} {arg.value}")
        self.record("all file paths relative", not bad, "; ".join(bad[:5]))

    def check_api(self) -> None:
        import importlib

        module = importlib.import_module(f"agent_code.{self.ship_name}.callbacks")
        ok = True
        for name, arity in REQUIRED_CALLBACKS.items():
            fn = getattr(module, name, None)
            if fn is None:
                ok = self.record(f"callbacks.{name} defined", False)
                continue
            got = len(inspect.signature(fn).parameters)
            ok &= self.record(f"callbacks.{name} takes {arity} args", got == arity, f"got {got}")
        train_path = self.ship_dir / "train.py"
        if train_path.exists():
            train = importlib.import_module(f"agent_code.{self.ship_name}.train")
            for name, arity in REQUIRED_TRAIN.items():
                fn = getattr(train, name, None)
                if fn is None:
                    self.record(f"train.{name} defined", False)
                    continue
                got = len(inspect.signature(fn).parameters)
                self.record(f"train.{name} takes {arity} args", got == arity, f"got {got}")
        if self._imports_torch():
            # torch otherwise spawns one thread per core and oversubscribes the
            # single tournament thread.
            joined = "\n".join(p.read_text() for p in self.ship_dir.rglob("*.py"))
            self.record("torch.set_num_threads(1) called", "set_num_threads(1)" in joined)
        return ok

    def _imports_torch(self) -> bool:
        for path in self.ship_dir.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "torch"
                                                        for a in node.names):
                    return True
                if (isinstance(node, ast.ImportFrom) and node.module
                        and node.module.split(".")[0] == "torch"):
                    return True
        return False

    def check_without_train_module(self) -> None:
        """The graders never import train.py; the agent must not need it."""
        stash = self.ship_dir / "train.py"
        moved = None
        if stash.exists():
            moved = stash.with_suffix(".py.stashed")
            stash.rename(moved)
        try:
            ok, detail = self.grader_simulation(rounds=2)
            self.record("runs with train.py absent", ok, detail)
        finally:
            if moved:
                moved.rename(stash)

    # -- dynamic checks ---------------------------------------------------
    def grader_simulation(self, rounds: int = 3) -> tuple[bool, str]:
        """Exactly what the graders run: train=False against three random agents."""
        cmd = [sys.executable, "main.py", "play", "--no-gui", "--n-rounds", str(rounds),
               "--agents", self.ship_name, "random_agent", "random_agent", "random_agent"]
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return False, proc.stderr.strip().splitlines()[-1] if proc.stderr else "non-zero exit"
        log = REPO / "logs" / "game.log"
        if not log.exists():
            return False, "logs/game.log not written"
        text = log.read_text()
        if f"Agent <{self.ship_name}> chose action ERROR" in text:
            return False, "agent raised (ERROR action)"
        return True, f"{rounds} rounds, no errors"

    def check_grader_simulation(self) -> None:
        ok, detail = self.grader_simulation()
        self.record("grader simulation (vs 3 random_agent)", ok, detail)

    def check_tournament_strength(self) -> None:
        from training.evaluate import evaluate, format_table, summarise

        records = evaluate(self.ship_name, ["rule_based_agent"] * 3, scenario="classic",
                           rounds=self.rounds, seed_base=20_000, workers=24, strict=True)
        agg = summarise(records)
        print(format_table(f"  {self.agent} vs 3 x rule_based_agent", agg))
        self.record(f"mean think time < {MAX_MEAN_MS} ms",
                    agg["think_ms_mean"] < MAX_MEAN_MS, f"{agg['think_ms_mean']:.2f} ms")
        self.record(f"p99 think time < {MAX_P99_MS} ms",
                    agg["think_ms_p99"] < MAX_P99_MS, f"{agg['think_ms_p99']:.2f} ms")
        self.record("no think-time timeouts", agg["think_timeouts"] == 0,
                    str(agg["think_timeouts"]))
        self.aggregate = agg

    # -- packaging --------------------------------------------------------
    def write_requirements(self) -> None:
        third_party = set()
        stdlib = set(sys.stdlib_module_names)
        for path in sorted(self.ship_dir.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                for name in names:
                    root = name.split(".")[0]
                    if root not in stdlib and root not in ("lib", "agent_code"):
                        third_party.add(root)
        lines = []
        for name in sorted(third_party):
            try:
                import importlib.metadata as md

                lines.append(f"{name}=={md.version(name)}")
            except Exception:
                lines.append(name)
        (AGENT_CODE / self.agent / "requirements.txt").write_text("\n".join(lines) + "\n")
        self.record("requirements.txt written", True, ", ".join(lines))

    def build_zip(self) -> Path:
        out = REPO / "build" / "final-project-agent-code.zip"
        out.parent.mkdir(parents=True, exist_ok=True)
        src = AGENT_CODE / self.agent
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(src.rglob("*")):
                if path.is_dir() or "__pycache__" in path.parts:
                    continue
                if path.name in ("USES_LIB",) or path.suffix == ".log":
                    continue
                if "logs" in path.relative_to(src).parts:
                    continue
                zf.write(path, Path(self.agent) / path.relative_to(src))
        self.record("zip built", True, f"{out} ({out.stat().st_size / 1e6:.2f} MB)")
        return out

    # -- driver -----------------------------------------------------------
    def run(self, do_zip: bool) -> int:
        print(f"checking agent_code/{self.agent} (shipped as {self.ship_name})")
        self.stage()
        try:
            self.check_files()
            self.check_source()
            self.check_relative_paths()
            self.check_api()
            self.check_grader_simulation()
            self.check_without_train_module()
            self.check_tournament_strength()
            self.write_requirements()
            if do_zip:
                self.build_zip()
        finally:
            self.cleanup()
        failed = [r for r in self.results if not r.ok]
        print(f"\n{len(self.results) - len(failed)}/{len(self.results)} checks passed")
        for r in failed:
            print(f"  FAILED: {r.name} {r.detail}")
        return 1 if failed else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--zip", action="store_true")
    p.add_argument("--out", default=None, help="write the metric summary here")
    args = p.parse_args(argv)
    checker = Checker(args.agent, args.rounds)
    code = checker.run(args.zip)
    if args.out and hasattr(checker, "aggregate"):
        Path(args.out).write_text(json.dumps(checker.aggregate, indent=1, default=float))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
