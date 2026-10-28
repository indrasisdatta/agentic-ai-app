# Agentic AI App

1. Create a virtual environment     
```bash python -m venv myvenv```

This project has old dependencies, so it's recommended to use Python 3.12 version.
Remove existing myvenv (for conflicting dependencies)

```bash 
Remove-Item -Recurse -Force myvenv # Windows PowerShell
py -3.12 -m venv myvenv

```

2. Activate environment (choose one based on your OS)
```bash myvenv/Scripts/activate.bat      # Windows CMD
.\myvenv\Scripts\Activate.ps1    # Windows PowerShell
python --version
```

3. Install dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```