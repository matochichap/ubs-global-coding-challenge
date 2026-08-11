import base64
import io
import json
import math
import operator
import re
from typing import Any

from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="UBS Global Coding Challenge Solver")

PRIORITY_MAP = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}
DEFAULT_PRIORITY = 2
MAX_RESPONSE_CHARS = 4500


class SolveRequest(BaseModel):
    payload: str


class SolveResponse(BaseModel):
    adaptOutput: dict[str, Any]
    sloOutput: dict[str, Any]


class EventPayload(BaseModel):
    payload: dict[str, Any] | None = None


class MCPRequest(BaseModel):
    jsonrpc: str
    id: Any = None
    method: str
    params: dict[str, Any] | None = None


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


def truncate_text(value: str) -> str:
    if len(value) <= MAX_RESPONSE_CHARS:
        return value
    return value[:MAX_RESPONSE_CHARS]


def get_name() -> str:
    return "solver-bot"


def do_arithmetic(expression: str) -> float | int:
    tokens = re.findall(r"\d+(?:\.\d+)?|[+\-*/()]", expression.replace(" ", ""))
    if "".join(tokens) != expression.replace(" ", ""):
        raise ValueError("Invalid expression")

    precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
    operations = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
    }

    output: list[float] = []
    ops: list[str] = []

    def apply_op() -> None:
        op = ops.pop()
        right = output.pop()
        left = output.pop()
        output.append(operations[op](left, right))

    for token in tokens:
        if token.replace(".", "", 1).isdigit():
            output.append(float(token))
            continue
        if token == "(":
            ops.append(token)
            continue
        if token == ")":
            while ops and ops[-1] != "(":
                apply_op()
            if not ops:
                raise ValueError("Mismatched parentheses")
            ops.pop()
            continue
        while ops and ops[-1] != "(" and precedence[ops[-1]] >= precedence[token]:
            apply_op()
        ops.append(token)

    while ops:
        if ops[-1] == "(":
            raise ValueError("Mismatched parentheses")
        apply_op()

    result = output[0]
    if float(result).is_integer():
        return int(result)
    return result


def identify_shapes(image_base64: str) -> dict[str, Any]:
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert("L")
    pixels = image.load()
    width, height = image.size

    visited = [[False for _ in range(width)] for _ in range(height)]
    counts = {"rectangle": 0, "triangle": 0, "circle": 0}

    def neighbors(x: int, y: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                out.append((nx, ny))
        return out

    def is_shape_pixel(x: int, y: int) -> bool:
        return pixels[x, y] < 200

    for y in range(height):
        for x in range(width):
            if visited[y][x] or not is_shape_pixel(x, y):
                continue

            stack = [(x, y)]
            visited[y][x] = True
            component: list[tuple[int, int]] = []

            while stack:
                cx, cy = stack.pop()
                component.append((cx, cy))
                for nx, ny in neighbors(cx, cy):
                    if not visited[ny][nx] and is_shape_pixel(nx, ny):
                        visited[ny][nx] = True
                        stack.append((nx, ny))

            if len(component) < 20:
                continue

            xs = [p[0] for p in component]
            ys = [p[1] for p in component]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            box_area = (max_x - min_x + 1) * (max_y - min_y + 1)
            fill_ratio = len(component) / box_area

            if fill_ratio > 0.72:
                counts["rectangle"] += 1
            elif fill_ratio < 0.58:
                counts["triangle"] += 1
            else:
                counts["circle"] += 1

    return counts


def mcp_error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def mcp_result(req_id: Any, result: Any) -> dict[str, Any]:
    if isinstance(result, str):
        result = truncate_text(result)
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


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


# -----------------------------
# Endpoint execution section
# -----------------------------
@app.post("/event")
def event_logger(payload: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "received": True}


@app.post("/mcp")
def mcp_endpoint(req: MCPRequest) -> dict[str, Any]:
    if req.jsonrpc != "2.0":
        return mcp_error(req.id, -32600, "Invalid Request")

    params = req.params or {}

    if req.method == "tools/list":
        tools = [
            {
                "name": "get_name",
                "description": "Returns the child's configured name as a string.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "do_arithmetic",
                "description": "Evaluates basic arithmetic operations (+, -, *, /).",
                "inputSchema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "identify_shapes",
                "description": "Identifies shape counts from a base64 PNG image.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"image_base64": {"type": "string"}},
                    "required": ["image_base64"],
                    "additionalProperties": False,
                },
            },
        ]
        return mcp_result(req.id, {"tools": tools})

    if req.method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        try:
            if tool_name == "get_name":
                output = get_name()
            elif tool_name == "do_arithmetic":
                output = do_arithmetic(arguments["expression"])
            elif tool_name == "identify_shapes":
                output = identify_shapes(arguments["image_base64"])
            else:
                return mcp_error(req.id, -32601, "Tool not found")
        except Exception as exc:
            return mcp_error(req.id, -32000, str(exc))

        return mcp_result(req.id, output)

    return mcp_error(req.id, -32601, "Method not found")