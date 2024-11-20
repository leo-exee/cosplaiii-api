# FastAPI Project

## Prerequisites

- [Python 3.10](https://www.python.org/downloads/release/python-3100/)
- [pip](https://pip.pypa.io/en/stable/installation/)
- [venv](https://docs.python.org/3/library/venv.html)

## Environment Setup

1. Create a virtual environment:
```bash
python3.10 -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Launch

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Documentations

- [FastAPI Official Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)