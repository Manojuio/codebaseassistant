# rag-service/app/rag/code/clone.py

from pathlib import Path
import subprocess
import hashlib

CLONE_DIR = Path("cloned_repos")
CLONE_DIR.mkdir(exist_ok=True)


def _repo_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:8]


def clone_repo(url: str) -> str:
    if not url.startswith("http"):
        raise ValueError("Invalid repository URL")

    repo_hash = _repo_hash(url)
    parts = url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError("Invalid repository URL format")

    repo_name = "_".join(parts[-2:])
    target_path = CLONE_DIR / f"{repo_name}_{repo_hash}"

    # If already exists → pull latest
    if target_path.exists():
        subprocess.run(
            ["git", "-C", str(target_path), "pull"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return str(target_path)

    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise Exception(result.stderr.strip())

    return str(target_path)