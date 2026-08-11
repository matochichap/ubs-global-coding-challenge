import base64
import io
import json
import math
import operator
import re
from typing import Any

from fastapi import FastAPI
from fastmcp import FastMCP
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="UBS Global Coding Challenge Solver")
mcp = FastMCP("solver-bot", app)

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


class MoveRequest(BaseModel):
    phase: int
    leg_number: int
    total_legs: int
    table_rule: str
    hand_number: int
    round: str
    your_number: int
    community_number: int | None
    pot: int
    to_call: int
    min_raise_to: int | None
    max_raise_to: int | None
    legal_actions: list[str]
    your_stack: int
    your_seat: int
    button_seat: int


class MoveResponse(BaseModel):
    action: str
    amount: int | None = None


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


def get_pre_reveal_action(your_number: int, to_call: int, min_raise_to: int, legal_actions: list[str]) -> tuple[str, int | None]:
    if your_number >= 11:
        if to_call == 0 and "raise" in legal_actions:
            return ("raise", min_raise_to)
        if to_call <= 10 and "call" in legal_actions:
            return ("call", None)
        return ("fold", None)
    else:
        if "check" in legal_actions:
            return ("check", None)
        return ("fold", None)


def get_post_reveal_action(your_number: int, community_number: int | None, to_call: int, max_raise_to: int, legal_actions: list[str]) -> tuple[str, int | None]:
    if your_number == community_number:
        if "raise" in legal_actions:
            return ("raise", max_raise_to)
        if "bet" in legal_actions:
            return ("bet", max_raise_to)
        return ("call", None)
    
    if your_number >= 11:
        if "check" in legal_actions:
            return ("check", None)
        if to_call <= 15 and "call" in legal_actions:
            return ("call", None)
        return ("fold", None)
    
    if "check" in legal_actions:
        return ("check", None)
    return ("fold", None)


def apply_fallback_chain(action: str, legal_actions: list[str]) -> str:
    if action in legal_actions:
        return action
    if "check" in legal_actions:
        return "check"
    if "call" in legal_actions:
        return "call"
    return "fold"


def play_standard(req: MoveRequest) -> tuple[str, int | None]:
    """Phase 2 Standard rules - same as Phase 1 TAG strategy."""
    if req.round == "pre_reveal":
        return get_pre_reveal_action(req.your_number, req.to_call, req.min_raise_to, req.legal_actions)
    else:
        return get_post_reveal_action(req.your_number, req.community_number, req.to_call, req.max_raise_to, req.legal_actions)


def exploit_standard(req: MoveRequest) -> tuple[str, int | None]:
    """Exploit Standard: Looser range + positional steals."""
    if req.round == "pre_reveal":
        # 11+: Raise or call anything up to 15
        if req.your_number >= 11:
            if req.to_call == 0 and "raise" in req.legal_actions:
                amount = max(req.min_raise_to or 1, 6)
                return ("raise", amount)
            if req.to_call <= 15 and "call" in req.legal_actions:
                return ("call", None)
            return ("fold", None)
        
        # 7-10: Call small bets, check if free
        if req.your_number >= 7:
            if req.to_call <= 5 and "call" in req.legal_actions:
                return ("call", None)
            if "check" in req.legal_actions:
                return ("check", None)
            return ("fold", None)
        
        # ≤6: Check if free, fold if bet
        if "check" in req.legal_actions:
            return ("check", None)
        return ("fold", None)
    else:
        # Post-reveal
        # Pair: All-in
        if req.your_number == req.community_number:
            if "raise" in req.legal_actions:
                return ("raise", req.max_raise_to)
            if "bet" in req.legal_actions:
                return ("bet", req.max_raise_to)
            return ("call", None)
        
        # Positional Steal: Act last (your_seat != button_seat) with no pair and free hand
        if (req.your_number < 11 and req.to_call == 0 and 
            req.your_seat != req.button_seat):
            if "raise" in req.legal_actions:
                amount = max(req.min_raise_to or 1, 10)
                return ("raise", amount)
            if "bet" in req.legal_actions:
                amount = max(req.min_raise_to or 1, 10)
                return ("bet", amount)
        
        # High Card (10-13): Call moderate bets
        if req.your_number >= 10:
            if "check" in req.legal_actions:
                return ("check", None)
            if req.to_call <= 20 and "call" in req.legal_actions:
                return ("call", None)
            return ("fold", None)
        
        # Miss: Check or fold
        if "check" in req.legal_actions:
            return ("check", None)
        return ("fold", None)


def play_low_ball(req: MoveRequest) -> tuple[str, int | None]:
    """Low Ball: lower numbers win, pairs are death traps."""
    if req.round == "pre_reveal":
        # Low (1-3): aggressive
        if req.your_number <= 3:
            if req.to_call == 0 and "raise" in req.legal_actions:
                return ("raise", req.min_raise_to)
            if req.to_call <= 10 and "call" in req.legal_actions:
                return ("call", None)
            return ("fold", None)
        # High (4+): avoid
        if "check" in req.legal_actions:
            return ("check", None)
        return ("fold", None)
    else:
        # Post-reveal: pair is worst hand
        if req.your_number == req.community_number:
            # DEATH TRAP: fold to any bet
            if req.to_call > 0:
                return ("fold", None)
            # Check if free
            if "check" in req.legal_actions:
                return ("check", None)
            return ("fold", None)
        
        # The Nuts: 1-3 with no pair
        if req.your_number <= 3:
            if "raise" in req.legal_actions:
                return ("raise", req.max_raise_to)
            if "bet" in req.legal_actions:
                return ("bet", req.max_raise_to)
            return ("call", None)
        
        # Miss: check or fold
        if "check" in req.legal_actions:
            return ("check", None)
        return ("fold", None)


def play_wild_seven(req: MoveRequest) -> tuple[str, int | None]:
    """Wild Seven: 7 acts as guaranteed pair."""
    if req.round == "pre_reveal":
        # 7 is a monster
        if req.your_number == 7:
            if "raise" in req.legal_actions and req.to_call == 0:
                return ("raise", req.min_raise_to)
            if req.to_call <= 40 and "call" in req.legal_actions:
                return ("call", None)
            return ("fold", None)
        
        # Premium (11+)
        if req.your_number >= 11:
            if req.to_call <= 10 and "call" in req.legal_actions:
                return ("call", None)
            if "check" in req.legal_actions:
                return ("check", None)
            return ("fold", None)
        
        # Other: check or fold
        if "check" in req.legal_actions:
            return ("check", None)
        return ("fold", None)
    else:
        # Post-reveal: 7 or pair is strong
        if req.your_number == 7 or req.your_number == req.community_number:
            # Raise ~50% of stack
            if "raise" in req.legal_actions:
                amount = round(req.your_stack * 0.5)
                amount = max(req.min_raise_to or 0, min(amount, req.max_raise_to or amount))
                return ("raise", amount)
            if "bet" in req.legal_actions:
                amount = round(req.your_stack * 0.5)
                amount = max(req.min_raise_to or 0, min(amount, req.max_raise_to or amount))
                return ("bet", amount)
            return ("call", None)
        
        # High card (11+)
        if req.your_number >= 11:
            if "check" in req.legal_actions:
                return ("check", None)
            if req.to_call <= 15 and "call" in req.legal_actions:
                return ("call", None)
            return ("fold", None)
        
        # Miss
        if "check" in req.legal_actions:
            return ("check", None)
        return ("fold", None)


def play_pair_bounty(req: MoveRequest) -> tuple[str, int | None]:
    """Pair Bounty: pairs win +5 chips, so minimize aggression to get to showdown."""
    if req.round == "pre_reveal":
        # Wider net: 8+ can call small bets
        if req.your_number >= 8:
            if req.to_call <= 5 and "call" in req.legal_actions:
                return ("call", None)
            if "check" in req.legal_actions:
                return ("check", None)
            return ("fold", None)
        
        # Lower: check or fold
        if "check" in req.legal_actions:
            return ("check", None)
        return ("fold", None)
    else:
        # Post-reveal
        if req.your_number == req.community_number:
            # Pair: min-bet to induce calls (don't scare them)
            if req.to_call == 0:
                if "bet" in req.legal_actions:
                    return ("bet", req.min_raise_to)
                if "check" in req.legal_actions:
                    return ("check", None)
            # If they bet, just call to reach showdown
            if "call" in req.legal_actions:
                return ("call", None)
            return ("fold", None)
        
        # High card (11+)
        if req.your_number >= 11:
            if "check" in req.legal_actions:
                return ("check", None)
            if req.to_call <= 15 and "call" in req.legal_actions:
                return ("call", None)
            return ("fold", None)
        
        # Miss
        if "check" in req.legal_actions:
            return ("check", None)
        return ("fold", None)


def exploit_pair_bounty(req: MoveRequest) -> tuple[str, int | None]:
    """Exploit Pair Bounty: Trap maniac's bluffs by calling, never raising pairs."""
    if req.round == "pre_reveal":
        # 8+: Catch bluffs by calling large bets (up to 25)
        if req.your_number >= 8:
            if req.to_call <= 25 and "call" in req.legal_actions:
                return ("call", None)
            if "check" in req.legal_actions:
                return ("check", None)
            return ("fold", None)
        
        # ≤7: Check if free, fold if bet
        if "check" in req.legal_actions:
            return ("check", None)
        return ("fold", None)
    else:
        # Post-reveal
        # Pair: THE TRAP - never raise, check to induce bluff or call it
        if req.your_number == req.community_number:
            # If free, check to induce their bluff
            if req.to_call == 0 and "check" in req.legal_actions:
                return ("check", None)
            # If they bet, call it (snap off bluff)
            if "call" in req.legal_actions:
                return ("call", None)
            return ("fold", None)
        
        # High Card (11-13): Hero call their massive bluffs (up to 50 chips or all-in)
        if req.your_number >= 11:
            # Check if free
            if req.to_call == 0 and "check" in req.legal_actions:
                return ("check", None)
            # Call even large bets up to 50 chips
            if req.to_call <= 50 and "call" in req.legal_actions:
                return ("call", None)
            # If bet is all-in or huge, still call (trap)
            if "call" in req.legal_actions and req.to_call >= req.your_stack * 0.5:
                return ("call", None)
            # Otherwise fold to huge bets
            return ("fold", None)
        
        # Miss: Check or fold
        if "check" in req.legal_actions:
            return ("check", None)
        return ("fold", None)


def validate_action(action: str, amount: int | None, req: MoveRequest) -> tuple[str, int | None]:
    """Validate action legality and enforce boundaries."""
    # Check legality
    if action not in req.legal_actions:
        action = apply_fallback_chain(action, req.legal_actions)
    
    # Enforce amount boundaries for raise/bet
    if action in ("raise", "bet") and amount is not None:
        min_amount = req.min_raise_to if req.min_raise_to is not None else 0
        max_amount = req.max_raise_to if req.max_raise_to is not None else amount
        amount = max(min_amount, min(amount, max_amount))
    
    # Remove amount for fold/check/call
    if action in ("fold", "check", "call"):
        amount = None
    
    return action, amount


def calculate_move(req: MoveRequest) -> tuple[str, int | None]:
    """Route to appropriate brain based on table_rule."""
    if req.table_rule == "standard":
        return exploit_standard(req)
    elif req.table_rule == "low_ball":
        return play_low_ball(req)
    elif req.table_rule == "wild_seven":
        return play_wild_seven(req)
    elif req.table_rule == "pair_bounty":
        return exploit_pair_bounty(req)
    else:
        # Safe fallback
        if "check" in req.legal_actions:
            return ("check", None)
        return ("fold", None)


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


@app.post("/move", response_model=MoveResponse)
def move(req: MoveRequest) -> MoveResponse:
    action, amount = calculate_move(req)
    action, amount = validate_action(action, amount, req)
    
    return MoveResponse(action=action, amount=amount)