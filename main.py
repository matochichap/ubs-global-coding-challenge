import base64
import json
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="UBS Global Coding Challenge Solver")

PRIORITY_MAP = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}
DEFAULT_PRIORITY = 2


class SolveRequest(BaseModel):
    payload: str


class SolveResponse(BaseModel):
    adaptOutput: dict[str, Any]


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest) -> SolveResponse:
    decoded = json.loads(base64.b64decode(req.payload).decode("utf-8"))
    adapt_input = decoded.get("adaptInput", {})

    user = adapt_input.get("user", {})
    metadata = adapt_input.get("metadata", {})

    priority = PRIORITY_MAP.get(metadata.get("priority"), DEFAULT_PRIORITY)

    return SolveResponse(
        adaptOutput={
            "id": user.get("id"),
            "name": user.get("fullName"),
            "action": str(adapt_input.get("action", "")).lower(),
            "priority": priority,
        }
    )