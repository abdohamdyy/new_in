#!/usr/bin/env python3
"""
Generate secure secrets for HMAC_CLIENT_SECRETS.

Usage examples:
  - python gen_hmac_secrets.py duaya_index salamtk
  - python gen_hmac_secrets.py duaya_index salamtk --format url --bytes 32
  - python gen_hmac_secrets.py --write-dotenv   # prompts for clients, writes .env

Outputs:
  - JSON value for HMAC_CLIENT_SECRETS
  - Ready-to-use commands: PowerShell, Bash, Windows CMD
  - docker-compose snippet
  - Optionally appends to .env (with --write-dotenv [path], default .env)
"""

from __future__ import annotations
import argparse
import json
import os
import secrets
import sys
from typing import Dict, List


def generate_secret(num_bytes: int, fmt: str) -> str:
    if fmt == "hex":
        # 32 bytes -> 64 hex chars
        return secrets.token_hex(num_bytes)
    if fmt == "url":
        # URL-safe base64-ish string
        return secrets.token_urlsafe(num_bytes)
    raise ValueError("Unsupported format; use 'hex' or 'url'")


def build_mapping(clients: List[str], num_bytes: int, fmt: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for raw in clients:
        client = (raw or "").strip()
        if not client:
            continue
        if client in mapping:
            continue
        mapping[client] = generate_secret(num_bytes, fmt)
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HMAC_CLIENT_SECRETS JSON")
    parser.add_argument("clients", nargs="*", help="Client IDs (e.g., duaya_index salamtk)")
    parser.add_argument("--bytes", dest="num_bytes", type=int, default=32,
                        help="Number of random bytes per secret (default: 32)")
    parser.add_argument("--format", choices=["hex", "url"], default="hex",
                        help="Secret format (default: hex)")
    parser.add_argument("--write-dotenv", nargs="?", const=".env", default=None,
                        help="Append HMAC_CLIENT_SECRETS to .env (optional path, default .env)")
    args = parser.parse_args()

    clients: List[str] = args.clients
    if not clients:
        try:
            raw = input("Enter client IDs (comma-separated): ").strip()
        except EOFError:
            raw = ""
        if raw:
            clients = [c.strip() for c in raw.split(",") if c.strip()]

    if not clients:
        print("No clients provided. Example: python gen_hmac_secrets.py duaya_index salamtk", file=sys.stderr)
        return 1

    if args.num_bytes < 16 or args.num_bytes > 128:
        print("--bytes must be between 16 and 128", file=sys.stderr)
        return 1

    mapping = build_mapping(clients, args.num_bytes, args.format)

    # Compact JSON safe for env variables
    json_value = json.dumps(mapping, ensure_ascii=False, separators=(",", ":"))

    print("HMAC_CLIENT_SECRETS JSON:")
    print(json_value)
    print()

    # PowerShell
    print("PowerShell (current session):")
    print(f"$env:HMAC_CLIENT_SECRETS='{json_value}'")
    print()

    # Windows CMD (persistent via setx)
    print("Windows CMD (persistent for new sessions):")
    print(f"setx HMAC_CLIENT_SECRETS \"{json_value}\"")
    print()

    # Bash / zsh
    print("Bash/zsh:")
    print(f"export HMAC_CLIENT_SECRETS='{json_value}'")
    print()

    # docker-compose snippet
    print("docker-compose snippet:")
    print("environment:")
    print(f"  - HMAC_CLIENT_SECRETS={json_value}")
    print()

    # .env optional write
    if args.write_dotenv:
        dotenv_path = os.fspath(args.write_dotenv)
        try:
            with open(dotenv_path, "a", encoding="utf-8") as f:
                f.write(f"\nHMAC_CLIENT_SECRETS={json_value}\n")
            print(f"Appended to {dotenv_path}: HMAC_CLIENT_SECRETS")
        except Exception as e:
            print(f"Failed to write {dotenv_path}: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


