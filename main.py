import base64
import json
import math
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
    sloOutput: dict[str, Any]


def build_adapt_output(adapt_input: dict[str, Any]) -> dict[str, Any]:
    user = adapt_input.get("user", {})
    metadata = adapt_input.get("metadata", {})

    priority = PRIORITY_MAP.get(metadata.get("priority"), DEFAULT_PRIORITY)

    return {
        "id": user.get("id"),
        "name": user.get("fullName"),
        "action": str(adapt_input.get("action", "")).lower(),
        "priority": priority,
    }


def build_slo_output(decoded: dict[str, Any]) -> dict[str, Any]:
    heartbeats = decoded.get("heartbeats", [])
    slo_query = decoded.get("sloQuery", {})

    service = slo_query.get("service")
    since = slo_query.get("since")

    seen = set()
    rows = []
    for hb in heartbeats:
        if service is not None and hb.get("service") != service:
            continue
        if since is not None and hb.get("timestamp", -1) < since:
            continue
        key = (hb.get("service"), hb.get("timestamp"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(hb)

    if not rows:
        return {"availability": 0.0, "p95LatencyMs": 0}

    ok_count = sum(1 for r in rows if r.get("status") == "OK")
    availability = ok_count / len(rows)

    latencies = sorted(r.get("latencyMs", 0) for r in rows)
    rank = math.ceil(0.95 * len(latencies))
    p95 = latencies[rank - 1]

    return {"availability": availability, "p95LatencyMs": p95}


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest) -> SolveResponse:
    decoded = json.loads(base64.b64decode(req.payload).decode("utf-8"))

    return SolveResponse(
        adaptOutput=build_adapt_output(decoded.get("adaptInput", {})),
        sloOutput=build_slo_output(decoded),
    )