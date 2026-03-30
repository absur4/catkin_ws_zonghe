#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Iterable, Set

import grasp_only_vision_viser as vis


def _extract_port_from_argv(argv: Iterable[str], default: int = 8080) -> int:
    port = default
    for arg in argv:
        if arg.startswith("_visualization/port:="):
            value = arg.split(":=", 1)[1].strip()
        elif arg.startswith("visualization/port:="):
            value = arg.split(":=", 1)[1].strip()
        else:
            continue
        try:
            port = int(value)
        except ValueError:
            pass
    return port


def _pids_from_lsof(port: int) -> Set[int]:
    if shutil.which("lsof") is None:
        return set()
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    pids = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.add(int(line))
        except ValueError:
            continue
    return pids


def _pids_from_ss(port: int) -> Set[int]:
    if shutil.which("ss") is None:
        return set()
    result = subprocess.run(
        ["ss", "-lptn", f"sport = :{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    pids = set()
    for line in result.stdout.splitlines():
        if "pid=" not in line:
            continue
        for match in re.finditer(r"pid=(\d+)", line):
            try:
                pids.add(int(match.group(1)))
            except ValueError:
                continue
    return pids


def _kill_pids(pids: Set[int], timeout_s: float = 1.0) -> None:
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            print(f"[viser-clean] 无权限终止进程 pid={pid}")
    deadline = time.time() + timeout_s
    while time.time() < deadline and pids:
        alive = set()
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.add(pid)
            except ProcessLookupError:
                continue
        pids = alive
        time.sleep(0.05)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            print(f"[viser-clean] 无权限强制终止进程 pid={pid}")


def _cleanup_port(port: int) -> None:
    pids = _pids_from_lsof(port)
    if not pids:
        pids = _pids_from_ss(port)
    if pids:
        print(f"[viser-clean] 清理端口 {port} 的旧进程: {sorted(pids)}")
    _kill_pids(pids)


def main() -> None:
    port = _extract_port_from_argv(sys.argv[1:])
    _cleanup_port(port)
    vis.main()


if __name__ == "__main__":
    main()
