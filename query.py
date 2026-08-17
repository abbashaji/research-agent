"""
query.py
Query your Turso DB from PowerShell without needing the Turso CLI (which
requires WSL on Windows). Reads TURSO_DATABASE_URL / TURSO_AUTH_TOKEN from
the environment, same as agent.py.

Usage (PowerShell):
    python query.py "SELECT run_id, n_findings, n_new FROM runs ORDER BY started_at DESC LIMIT 5;"
    python query.py "SELECT tier, pass_type, title, url FROM findings WHERE topic='procedural building generator' LIMIT 10;"
"""
import os
import sys
import libsql_client

def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "SELECT ..."')
        sys.exit(1)
    url = os.environ.get("TURSO_DATABASE_URL")
    token = os.environ.get("TURSO_AUTH_TOKEN")
    if not (url and token):
        print("Set TURSO_DATABASE_URL and TURSO_AUTH_TOKEN first ($env:VAR = \"...\")")
        sys.exit(1)
    url = url.replace("libsql://", "https://").replace("wss://", "https://")
    client = libsql_client.create_client_sync(url=url, auth_token=token)
    result = client.execute(sys.argv[1])
    if result.columns:
        print(" | ".join(result.columns))
        print("-" * 60)
    for row in result.rows:
        print(" | ".join(str(v) for v in row))
    print(f"\n({len(result.rows)} rows)")

if __name__ == "__main__":
    main()
