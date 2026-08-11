import base64
import heapq
import io
import json
import math
import operator
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any

from fastapi import FastAPI
from fastmcp import FastMCP
from PIL import Image
from pydantic import BaseModel

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency during local setup
    tiktoken = None

app = FastAPI(title="UBS Global Coding Challenge Solver")
mcp = FastMCP("solver-bot")

STUDY_MATERIALS_BASE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com/study-materials"
GRAPH_BASE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com/graph"
TOKEN_BUDGET = 900
DEFAULT_MAX_PASSAGES = 8

_study_index: list[dict[str, Any]] | None = None
_study_index_signature: tuple[int, ...] | None = None

SEMANTIC_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "alignment": ("calibration", "calibrated", "recalibrated", "aligned", "realigned", "restore", "restored", "maintenance"),
    "aligned": ("calibration", "calibrated", "recalibrated", "alignment", "realigned", "restore", "restored"),
    "sensor": ("array", "hydrophone", "instrument", "detector", "probe", "monitor"),
    "grid": ("array", "network", "system", "matrix", "mesh"),
    "route": ("journey", "path", "hop", "hops", "edge", "adjacency", "toll"),
    "journey": ("route", "path", "hop", "hops", "edge", "toll", "adjacency"),
    "trial": ("protocol", "cohort", "dose", "dosage", "sample", "adverse", "placebo"),
    "station": ("habitat", "submersible", "pressure", "acclimation", "calibration", "maintenance"),
    "fare": ("cap", "ticket", "payment", "price", "charge", "policy"),
    "engine": ("turbine", "handbook", "torque", "lubrication", "diagnostic", "maintenance"),
    "growers": ("harvest", "crop", "yield", "cooperative", "seed", "irrigation"),
}

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
    """Truncates text to MAX_RESPONSE_CHARS."""
    if len(value) <= MAX_RESPONSE_CHARS:
        return value
    return value[:MAX_RESPONSE_CHARS]


def _fetch_json(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to fetch JSON from {url}") from exc


def _fetch_text(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            return response.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise ValueError(f"Failed to fetch text from {url}") from exc


def _normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _expand_query_tokens(query_tokens: list[str]) -> list[str]:
    expanded = list(query_tokens)
    for token in query_tokens:
        expanded.extend(SEMANTIC_EXPANSIONS.get(token, ()))
    return expanded


def _estimate_tokens(text: str) -> int:
    if tiktoken is not None:
        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    return max(1, math.ceil(len(_normalize_tokens(text)) * 1.35))


def _split_passages(text: str) -> list[str]:
    passages = []
    for chunk in re.split(r"\n\s*\n", text):
        cleaned = re.sub(r"\s+", " ", chunk).strip()
        if len(cleaned) >= 40:
            passages.append(cleaned)
    return passages


def _load_study_index() -> list[dict[str, Any]]:
    global _study_index, _study_index_signature

    summary = _fetch_json(STUDY_MATERIALS_BASE_URL)
    doc_ids = tuple(doc["id"] for doc in summary.get("documents", []))
    if _study_index is not None and _study_index_signature == doc_ids:
        return _study_index

    index: list[dict[str, Any]] = []
    documents = summary.get("documents", [])
    for doc in documents:
        doc_id = doc["id"]
        title = doc.get("title", f"Document {doc_id}")
        content = _fetch_text(f"{STUDY_MATERIALS_BASE_URL}/{doc_id}")
        for passage in _split_passages(content):
            index.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text": passage,
                    "tokens": _normalize_tokens(passage),
                    "token_count": _estimate_tokens(passage),
                }
            )

    _study_index = index
    _study_index_signature = doc_ids
    return index


def _score_passage(query_tokens: list[str], passage: dict[str, Any]) -> float:
    if not query_tokens:
        return 0.0

    passage_tokens = passage["tokens"]
    passage_counts = Counter(passage_tokens)
    query_counts = Counter(query_tokens)

    overlap = sum(min(query_counts[token], passage_counts.get(token, 0)) for token in query_counts)
    if overlap == 0:
        return 0.0

    length_penalty = 1.0 + math.log1p(max(1, len(passage_tokens)))
    title_tokens = set(_normalize_tokens(passage["title"]))
    title_boost = sum(2.0 for token in query_counts if token in title_tokens)

    phrase_boost = 0.0
    joined_query = " ".join(query_tokens[:8])
    if joined_query and joined_query in passage["text"].lower():
        phrase_boost += 4.0

    return (overlap * 10.0 + title_boost + phrase_boost) / length_penalty


def _select_passages(query: str, max_passages: int = DEFAULT_MAX_PASSAGES) -> list[str]:
    index = _load_study_index()
    query_tokens = _expand_query_tokens(_normalize_tokens(query))

    scored = sorted(
        index,
        key=lambda passage: (
            _score_passage(query_tokens, passage),
            -passage["token_count"],
        ),
        reverse=True,
    )

    selected: list[str] = []
    total_tokens = 0
    for passage in scored:
        if passage["token_count"] > TOKEN_BUDGET:
            continue
        if total_tokens + passage["token_count"] > TOKEN_BUDGET:
            continue
        if passage["text"] in selected:
            continue
        if _score_passage(query_tokens, passage) <= 0:
            continue
        selected.append(passage["text"])
        total_tokens += passage["token_count"]
        if len(selected) >= max_passages:
            break

    if not selected:
        fallback = scored[: min(max_passages, 3)]
        for passage in fallback:
            if total_tokens + passage["token_count"] > TOKEN_BUDGET:
                continue
            if passage["text"] in selected:
                continue
            selected.append(passage["text"])
            total_tokens += passage["token_count"]

    return selected


def _fetch_graph(map_id: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"map_id": map_id})
    return _fetch_json(f"{GRAPH_BASE_URL}?{query}")


def _parse_route_question(question: str) -> tuple[str, str, str, int | None]:
    map_match = re.search(r"map_id\s*[:=]\s*([^\s]+)", question, re.IGNORECASE)
    if not map_match:
        raise ValueError("Question does not include a map_id")

    route_match = re.search(r"from\s+(.+?)\s+to\s+(.+?)(?:\?|$)", question, re.IGNORECASE)
    if not route_match:
        raise ValueError("Question does not include a start and destination")

    hop_match = re.search(
        r"(?:hops?\s*(?:left|remaining)?|allowance(?:\s+of)?)\s*[:=]?\s*(\d+)",
        question,
        re.IGNORECASE,
    )

    start = route_match.group(1).strip().strip(".,")
    destination = route_match.group(2).strip().strip(".,")
    map_id = map_match.group(1).strip().strip(".,")
    hop_limit = int(hop_match.group(1)) if hop_match else None
    return start, destination, map_id, hop_limit


def _shortest_path(graph: dict[str, Any], start: str, destination: str, hop_limit: int | None = None) -> list[str]:
    adjacency = graph.get("adjacency", {})
    tolls = graph.get("tolls", {})

    if start == destination:
        return [start]

    node_set = set(adjacency) | set(tolls)
    for neighbors in adjacency.values():
        node_set.update(neighbors)

    max_edges = hop_limit if hop_limit is not None else max(0, len(node_set) - 1)
    if max_edges <= 0:
        raise ValueError("No hops available")

    start_state = (start, max_edges)
    queue: list[tuple[float, str, int]] = [(0.0, start, max_edges)]
    distances: dict[tuple[str, int], float] = {start_state: 0.0}
    previous: dict[tuple[str, int], tuple[str, int]] = {}
    best_destination_state: tuple[str, int] | None = None
    best_destination_cost = float("inf")

    while queue:
        cost, node, remaining = heapq.heappop(queue)
        state = (node, remaining)
        if cost != distances.get(state):
            continue

        if node == destination and cost < best_destination_cost:
            best_destination_cost = cost
            best_destination_state = state

        if remaining == 0:
            continue

        for neighbor, edge_cost in adjacency.get(node, {}).items():
            next_cost = cost + float(edge_cost) + float(tolls.get(neighbor, 0))
            next_state = (neighbor, remaining - 1)
            if next_cost < distances.get(next_state, float("inf")):
                distances[next_state] = next_cost
                previous[next_state] = state
                heapq.heappush(queue, (next_cost, neighbor, remaining - 1))

    if best_destination_state is None:
        raise ValueError(f"No route found from {start} to {destination}")

    path = [best_destination_state[0]]
    state = best_destination_state
    while state in previous:
        state = previous[state]
        path.append(state[0])
    path.reverse()
    return path


@mcp.tool()
def get_name() -> str:
    """Returns the child's configured name as a string."""
    return "solver-bot"


@mcp.tool()
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


@mcp.tool()
def identify_shapes(image_base64: str) -> dict[str, Any]:
    """Identifies shape counts from a base64 PNG image."""
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


@mcp.tool()
def search_study_materials(question: str, max_passages: int = DEFAULT_MAX_PASSAGES) -> list[str]:
    """Returns the most relevant passages from the study material corpus for a revision question."""
    return _select_passages(question, max_passages=max_passages)


@mcp.tool()
def next_city_hop(question: str) -> str:
    """Returns the next node label for a route question, honoring tolls and any hop allowance in the prompt."""
    start, destination, map_id, hop_limit = _parse_route_question(question)
    graph = _fetch_graph(map_id)
    path = _shortest_path(graph, start, destination, hop_limit=hop_limit)
    return path[0] if len(path) == 1 else path[1]


def truncate_text(value: str) -> str:
    """Truncates text to MAX_RESPONSE_CHARS."""
    if len(value) <= MAX_RESPONSE_CHARS:
        return value
    return value[:MAX_RESPONSE_CHARS]


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


mcp_app = mcp.http_app(path="/mcp")
app = FastAPI(
    title="UBS Global Coding Challenge Solver",
    routes=[*mcp_app.routes, *app.routes],
    lifespan=mcp_app.lifespan,
)


if __name__ == "__main__":
    mcp.run()


