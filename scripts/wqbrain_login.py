"""
One-time WQ Brain login. Saves a session token to ~/.wqbrain/session.json.

Usage:
  python scripts/wqbrain_login.py              # interactive
  python scripts/wqbrain_login.py --save-credentials  # save username+password for auto-reauth
  python scripts/wqbrain_login.py --whoami     # verify saved session
  python scripts/wqbrain_login.py --cookie XYZ # save a session cookie directly
  python scripts/wqbrain_login.py --logout     # delete the saved session
"""

import argparse
import getpass
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# Avoid the heavy memory_layer/__init__ chain (sentence-transformers).
import importlib.util


def _load_brain_api():
    spec = importlib.util.spec_from_file_location(
        "brain_api", BASE / "memory_layer" / "brain_api.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


brain_api = _load_brain_api()


def cmd_interactive():
    print("WQ Brain login\n")
    print("Choose an authentication method:")
    print("  1) Username + password (typed at this prompt; password not stored)")
    print("  2) Paste a session cookie copied from your browser DevTools")
    choice = input("\nChoice [1/2]: ").strip()

    if choice == "1":
        username = input("Username (email): ").strip()
        password = getpass.getpass("Password: ")
        try:
            session = brain_api.login_with_password(username, password)
        except brain_api.BrainAuthError as e:
            print(f"\nLogin failed: {e}", file=sys.stderr)
            sys.exit(1)
    elif choice == "2":
        print()
        print("In your browser:")
        print("  1. Log in at https://platform.worldquantbrain.com")
        print("  2. Open DevTools (F12) → Application/Storage → Cookies → api.worldquantbrain.com")
        print("  3. Copy the value of the session cookie (often named 't')")
        print()
        cookie_name = input("Cookie name [t]: ").strip() or "t"
        cookie_value = input("Cookie value: ").strip()
        if not cookie_value:
            print("Empty cookie value — aborting", file=sys.stderr)
            sys.exit(1)
        session = brain_api.login_with_cookie(cookie_value, cookie_name=cookie_name)
    else:
        print("Invalid choice", file=sys.stderr)
        sys.exit(1)

    client = brain_api.BrainAPIClient(session=session)
    client.save()
    print(f"\nSession saved to {brain_api.SESSION_FILE}")
    print("Verifying with /users/self ...")
    try:
        info = client.whoami()
        print(f"  ok — identity: {info}")
    except brain_api.BrainAuthError as e:
        print(f"  WARN: {e}", file=sys.stderr)
        print("  The cookie may be wrong, expired, or named differently.", file=sys.stderr)
        sys.exit(2)


def cmd_whoami():
    try:
        client = brain_api.BrainAPIClient.from_disk()
        info = client.whoami()
        print(info)
    except brain_api.BrainAuthError as e:
        print(f"Auth error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_cookie(cookie_value: str, cookie_name: str):
    session = brain_api.login_with_cookie(cookie_value, cookie_name=cookie_name)
    client = brain_api.BrainAPIClient(session=session)
    client.save()
    print(f"Session saved to {brain_api.SESSION_FILE}")
    try:
        info = client.whoami()
        print(f"Verified — identity: {info}")
    except brain_api.BrainAuthError as e:
        print(f"WARN: {e}", file=sys.stderr)
        sys.exit(2)


def cmd_save_credentials():
    """Save username + password to ~/.wqbrain/credentials.json for auto-reauth."""
    print("Save credentials for automatic re-login when session expires.\n")
    username = input("Username (email): ").strip()
    if not username:
        print("Empty username — aborting", file=sys.stderr)
        sys.exit(1)
    password = getpass.getpass("Password: ")
    if not password:
        print("Empty password — aborting", file=sys.stderr)
        sys.exit(1)

    # Verify credentials work before saving
    print("Verifying credentials ...")
    try:
        session = brain_api.login_with_password(username, password)
    except brain_api.BrainAuthError as e:
        print(f"Login failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Save credentials
    creds_file = brain_api.CREDENTIALS_FILE
    creds_file.parent.mkdir(parents=True, exist_ok=True)
    creds_file.write_text(
        __import__("json").dumps({"username": username, "password": password}, indent=2),
        encoding="utf-8",
    )
    # Restrict permissions on non-Windows (chmod 600)
    try:
        import stat
        creds_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass  # Windows doesn't support chmod — acceptable

    # Also save the fresh session
    client = brain_api.BrainAPIClient(session=session)
    client.save()
    print(f"Credentials saved to {creds_file}")
    print(f"Session saved to {brain_api.SESSION_FILE}")
    print("Auto-reauth is now active — sessions will renew automatically on expiry.")


def cmd_logout():
    p = brain_api.SESSION_FILE
    if p.exists():
        p.unlink()
        print(f"Deleted {p}")
    else:
        print(f"No saved session at {p}")


def main():
    ap = argparse.ArgumentParser(description="WQ Brain login helper")
    ap.add_argument("--save-credentials", action="store_true",
                    help="Save username+password for automatic re-login on expiry")
    ap.add_argument("--whoami", action="store_true", help="Verify the saved session")
    ap.add_argument("--cookie", help="Set session cookie value directly")
    ap.add_argument("--cookie-name", default="t", help="Session cookie name (default: t)")
    ap.add_argument("--logout", action="store_true", help="Delete saved session")
    args = ap.parse_args()

    if args.save_credentials:
        cmd_save_credentials()
    elif args.logout:
        cmd_logout()
    elif args.whoami:
        cmd_whoami()
    elif args.cookie:
        cmd_cookie(args.cookie, args.cookie_name)
    else:
        cmd_interactive()


if __name__ == "__main__":
    main()
