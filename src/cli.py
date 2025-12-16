from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> int:
    res = subprocess.run(cmd)
    return res.returncode


def main() -> int:
    p = argparse.ArgumentParser(prog="mih", description="Media Intelligence Hub CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("bootstrap", help="Start infra + init schema (make bootstrap)")
    sub.add_parser("smoke", help="Full local sanity check (make smoke)")

    p_etl = sub.add_parser("etl", help="Run ETL")
    p_etl.add_argument("--latest", action="store_true", help="Use latest silver file (make etl-latest)")
    p_etl.add_argument("--in", dest="in_path", type=str, help="Silver file path for make etl IN=...")

    p_rep = sub.add_parser("report", help="Generate SQL + Markdown reports (make reports)")
    p_rep.add_argument("--last-hours", type=int, default=None)
    p_rep.add_argument("--from", dest="from_dt", type=str, default=None)
    p_rep.add_argument("--to", dest="to_dt", type=str, default=None)

    args = p.parse_args()

    if args.cmd == "bootstrap":
        return run(["make", "bootstrap"])

    if args.cmd == "smoke":
        return run(["make", "smoke"])

    if args.cmd == "etl":
        if args.latest:
            return run(["make", "etl-latest"])
        if args.in_path:
            return run(["make", "etl", f"IN={args.in_path}"])
        print("mih etl: use --latest or --in <silver_file>", file=sys.stderr)
        return 2

    if args.cmd == "report":
        cmd = ["make", "reports"]
        if args.last_hours is not None:
            cmd.append(f"LAST_HOURS={args.last_hours}")
        if args.from_dt is not None:
            cmd.append(f'FROM={args.from_dt}')
        if args.to_dt is not None:
            cmd.append(f'TO={args.to_dt}')
        return run(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())