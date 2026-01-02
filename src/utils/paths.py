from pathlib import Path

def project_root() -> Path:
    try:
        current = Path(__file__).resolve().parent
    except NameError:
        current = Path.cwd()
    
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    
    raise RuntimeError("Project root not found (pyproject.toml missing)")