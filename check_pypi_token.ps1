# Verify a PyPI API token without uploading any package.
#
# Why Python instead of pure PowerShell:
#   PyPI's CDN rejects POST from .NET's HTTP stack (Invoke-WebRequest /
#   HttpWebRequest) with 405 before the auth check runs. Calling Python's
#   `requests` library — what twine itself uses — replicates twine's wire
#   format so PyPI reaches the auth check and returns a meaningful status.
#
# Why a temp file instead of `python -c $probe`:
#   PowerShell 5.1 strips quotes from arguments to native executables,
#   mangling Python source passed via -c.
#
# Why an env var instead of stdin pipe:
#   PowerShell's pipe to a native exe transcodes via the console code page,
#   which on non-en-US Windows can prepend BOM/wide bytes that requests'
#   latin-1 Basic-Auth encoder rejects. Env vars travel in UTF-16 and are
#   exposed to Python through os.environ cleanly.
#
# Test: POST to https://upload.pypi.org/legacy/ with valid Basic auth and a
# multipart form containing only the ":action=file_upload" sentinel.
#   400 : auth OK, fields incomplete  -> token VALID
#   403 : auth rejected               -> token INVALID
#   other : printed verbatim for diagnosis
#
# The token is read with Read-Host -AsSecureString, kept only in this
# process's memory + the spawned python child, never written to disk.

$python = "c:/Y/anaconda3/python.exe"

$tokenSecure = Read-Host "Paste PyPI token (input is hidden)" -AsSecureString
$token = [System.Net.NetworkCredential]::new("", $tokenSecure).Password

$probe = @'
import os
import sys
import requests

token = os.environ.get("PYPI_TOKEN_PROBE", "").strip()
if not token:
    print("ERROR: empty token")
    sys.exit(2)

try:
    r = requests.post(
        "https://upload.pypi.org/legacy/",
        auth=("__token__", token),
        files={
            ":action": (None, "file_upload"),
            "protocol_version": (None, "1"),
        },
        timeout=20,
    )
except requests.RequestException as e:
    print(f"NETWORK ERROR: {e}")
    sys.exit(3)

code = r.status_code
body = (r.text or "").strip().splitlines()
first = body[0] if body else ""

if code == 400:
    print(f"VALID   - token authenticated (HTTP 400, expected: {first})")
elif code == 403:
    print(f"INVALID - auth rejected (HTTP 403): {first}")
    print("  Causes:")
    print("   * token mistyped or partially copied (must include 'pypi-' prefix)")
    print("   * token revoked or replaced")
    print("   * project-scoped token used for a different project")
    print("   * brand-new package with project-scoped token (entire-account required for first upload)")
else:
    print(f"UNEXPECTED HTTP {code}: {first}")
    print("--- response body (first 800 chars) ---")
    print((r.text or "")[:800])
'@

$tmpFile = Join-Path $env:TEMP ("check_pypi_token_" + [Guid]::NewGuid().ToString() + ".py")
Set-Content -Path $tmpFile -Value $probe -Encoding UTF8
$env:PYPI_TOKEN_PROBE = $token
try {
    & $python $tmpFile
} finally {
    Remove-Item Env:PYPI_TOKEN_PROBE -ErrorAction SilentlyContinue
    Remove-Item $tmpFile -Force -ErrorAction SilentlyContinue
}
