#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
from pathlib import Path

SRC = Path("bot_multi.py")
BAK = Path("bot_multi.backup.py")

def main():
    if not SRC.exists():
        raise SystemExit("bot_multi.py not found in current directory.")
    original = SRC.read_text(encoding="utf-8")

    patched = original

    # 1) Fix scheduler timestamp logs (seconds-based 'next_close')
    patched = re.sub(
        r"datetime\.utcfromtimestamp\(\s*next_close\s*\)",
        "datetime.fromtimestamp(next_close, datetime.UTC)",
        patched
    )

    # 2) Fix bar timestamp logs that use milliseconds (self.last_bar_ts)
    patched = re.sub(
        r"datetime\.utcfromtimestamp\(\s*self\.last_bar_ts\s*\)",
        "datetime.fromtimestamp(self.last_bar_ts/1000, datetime.UTC)",
        patched
    )

    # 3) Optional: any remaining utcfromtimestamp() -> fromtimestamp(..., datetime.UTC)
    #    (This keeps the same argument; for ms-based ones, devs can update similarly)
    patched = re.sub(
        r"datetime\.utcfromtimestamp\(",
        "datetime.fromtimestamp(",
        patched
    )

    if patched == original:
        print("[INFO] No changes made (patterns not found).")
        return

    # Backup then write
    if not BAK.exists():
        BAK.write_text(original, encoding="utf-8")
        print("[INFO] Backup saved to", BAK)
    SRC.write_text(patched, encoding="utf-8")
    print("[OK] Patched bot_multi.py")
    print("-> Re-deploy your service and watch for [BAR] and [SIGNAL] logs.")

if __name__ == "__main__":
    main()