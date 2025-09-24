#!/usr/bin/env python3
"""
Simple CLI to compute HMAC-SHA256 signature in hex.

Usage examples:
  - python hmac_hex.py --secret SECRET --base "method=GET&path=/search&ts=1690000000&payload={\"q\":\"بانادول\"}"
  - python hmac_hex.py -s SECRET -m GET -p /search -t now -d '{"q":"بانادول"}'

If --ts is "now" or omitted when using --method/--path, current Unix seconds are used.
Optionally prints headers when --client is provided.
"""

from __future__ import annotations
import argparse
import hmac
import hashlib
import os
import sys
import time


def compute_hmac_hex(secret: str, base: str) -> str:
    return hmac.new(secret.encode("utf-8"), base.encode("utf-8"), hashlib.sha256).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute HMAC-SHA256 hex signature")
    parser.add_argument("--secret", "-s", help="Shared secret (or set HMAC_SECRET env)")
    parser.add_argument("--base", "-b", help="Full base string to sign")
    parser.add_argument("--method", "-m", help="HTTP method (e.g., GET, POST)")
    parser.add_argument("--path", "-p", help="Request path (e.g., /search)")
    parser.add_argument("--ts", "-t", help="Unix seconds timestamp or 'now' (default: now)")
    parser.add_argument("--payload", "-d", default="", help="Payload string (default empty)")
    parser.add_argument("--client", "-c", help="Optional client id to echo header lines")
    args = parser.parse_args()

    secret = args.secret or os.getenv("HMAC_SECRET")
    if not secret:
        print("Missing --secret (or set HMAC_SECRET env)", file=sys.stderr)
        return 1

    base = args.base
    if not base:
        if not args.method or not args.path:
            print("Either --base or (--method and --path) must be provided", file=sys.stderr)
            return 1
        ts = args.ts or "now"
        if ts == "now":
            ts = str(int(time.time()))
        base = f"method={args.method.upper()}&path={args.path}&ts={ts}&payload={args.payload}"
    sig = compute_hmac_hex(secret, base)

    print(sig)

    # Optional helper output
    if args.client or args.method:
        # Extract ts from base if present
        ts_value = None
        for part in base.split("&"):
            if part.startswith("ts="):
                ts_value = part[3:]
                break
        if ts_value and args.client:
            sys.stderr.write("\nSuggested headers:\n")
            sys.stderr.write(f"X-Client-Id: {args.client}\n")
            sys.stderr.write(f"X-Timestamp: {ts_value}\n")
            sys.stderr.write(f"X-Signature: {sig}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


