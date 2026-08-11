from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="UBS Global Coding Challenge Solver")


class SolveRequest(BaseModel):
    input: Any


class SolveResponse(BaseModel):
    result: Any


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve")
def solve():
    return {"status": "ok"}