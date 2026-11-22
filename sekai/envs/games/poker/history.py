"""PokerStars-style text hand histories.

The history is rendered from a `SeatView`, so it can only contain what that view may see:
pass the hand with `hero=seat` for a player's own history (their hole cards plus anything
shown at showdown), `reveal=True` once the hand is over to show both hands, or neither for
a spectator's history. Example:

```text
sekai Hand #7: Hold'em No Limit (50/100) - 2026/10/18 12:00:00 UTC
Table 'sekai' 2-max Seat #1 is the button
Seat 1: You (10000 in chips)
Seat 2: Agent (10000 in chips)
You: posts small blind 50
Agent: posts big blind 100
*** HOLE CARDS ***
Dealt to You [As Kd]
You: raises 200 to 300
Agent: calls 200
*** FLOP *** [Ah 7d 2c]
Agent: checks
You: bets 400
Agent: folds
Uncalled bet (400) returned to You
You collected 600 from pot
You: doesn't show hand
*** SUMMARY ***
Total pot 600 | Rake 0
Board [Ah 7d 2c]
Seat 1: You (button) (small blind) collected (600)
Seat 2: Agent (big blind) folded on the Flop
```
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence

from sekai.envs.games.poker.cards import card_to_str, format_cards
from sekai.envs.games.poker.holdem import HoldemHand, SeatView

_STREET_TITLES = {"flop": "FLOP", "turn": "TURN", "river": "RIVER"}
_FOLDED_ON = {
    "preflop": "before Flop",
    "flop": "on the Flop",
    "turn": "on the Turn",
    "river": "on the River",
}


def format_hand_history(
    hand: HoldemHand | SeatView,
    *,
    hand_no: int,
    names: Sequence[str] = ("Seat 1", "Seat 2"),
    table: str = "sekai",
    site: str = "sekai",
    hero: int | None = None,
    reveal: bool = False,
    timestamp: _dt.datetime | None = None,
) -> str:
    """The text hand history of `hand` (a `HoldemHand` or a `SeatView`).

    A hand in progress is rendered up to the current point, without a summary.

    Args:
        hand: The hand, or a view of it (exactly that view's knowledge is used).
        hand_no: The hand's number in the header.
        names: `names[s]` names seat `s` (displayed as "Seat s+1").
        table: The table's name.
        site: The name that opens the header (`"{site} Hand #{hand_no}: ..."`).
        hero: With a hand, whose hole cards appear ("Dealt to ...").
        reveal: With a hand, show both hands once it is complete.
        timestamp: The hand's time, by default now (in UTC).
    """
    if isinstance(hand, HoldemHand):
        if reveal and hand.is_complete:
            view = hand.reveal()
        elif hero is not None:
            view = hand.view(hero)
        else:
            view = hand.public_view()
    elif isinstance(hand, SeatView):
        view = hand
    else:
        raise TypeError(f"Expected a HoldemHand or SeatView, got {type(hand).__name__}.")
    if len(names) != 2:
        raise ValueError("names needs one name per seat.")
    ts = timestamp or _dt.datetime.now(_dt.timezone.utc)
    if ts.tzinfo is not None:
        ts = ts.astimezone(_dt.timezone.utc)

    lines = [
        f"{site} Hand #{hand_no}: Hold'em No Limit ({view.small_blind}/{view.big_blind}) - "
        f"{ts:%Y/%m/%d %H:%M:%S} UTC",
        f"Table '{table}' 2-max Seat #{view.button + 1} is the button",
    ]
    for s in (0, 1):
        lines.append(f"Seat {s + 1}: {names[s]} ({view.starting_stacks[s]} in chips)")

    street_bets = [0, 0]
    folded: dict[int, str] = {}
    collected = [0, 0]
    shown_any = False
    hole_header_done = False
    for ev in view.events:
        kind = ev["type"]
        if kind != "post" and not hole_header_done:
            lines.extend(_hole_lines(view, names))
            hole_header_done = True
        if kind == "post":
            street_bets[ev["seat"]] = ev["amount"]
            suffix = " and is all-in" if ev["all_in"] else ""
            lines.append(f"{names[ev['seat']]}: posts {ev['blind']} blind {ev['amount']}{suffix}")
        elif kind == "action":
            name = names[ev["seat"]]
            current = max(street_bets)
            suffix = " and is all-in" if ev["all_in"] else ""
            act = ev["kind"]
            if act == "fold":
                folded[ev["seat"]] = ev["street"]
                lines.append(f"{name}: folds")
            elif act == "check":
                lines.append(f"{name}: checks")
            elif act == "call":
                lines.append(f"{name}: calls {ev['added']}{suffix}")
            elif act == "bet":
                lines.append(f"{name}: bets {ev['to']}{suffix}")
            else:
                lines.append(f"{name}: raises {ev['to'] - current} to {ev['to']}{suffix}")
            street_bets[ev["seat"]] = ev["to"]
        elif kind == "deal":
            street_bets = [0, 0]
            title = _STREET_TITLES[ev["street"]]
            new = ev["cards"]
            before = _board_before(view, ev["street"])
            if before:
                lines.append(f"*** {title} *** [{' '.join(before)}] [{' '.join(new)}]")
            else:
                lines.append(f"*** {title} *** [{' '.join(new)}]")
        elif kind == "return":
            lines.append(f"Uncalled bet ({ev['amount']}) returned to {names[ev['seat']]}")
        elif kind == "showdown":
            if not shown_any:
                lines.append("*** SHOW DOWN ***")
                shown_any = True
            lines.append(f"{names[ev['seat']]}: shows [{' '.join(ev['cards'])}] ({ev['rank']})")
        elif kind == "win":
            collected[ev["seat"]] += ev["amount"]
            lines.append(f"{names[ev['seat']]} collected {ev['amount']} from pot")
            if view.result is not None and not view.result.showdown:
                lines.append(f"{names[ev['seat']]}: doesn't show hand")
    if not hole_header_done:
        lines.extend(_hole_lines(view, names))

    if view.result is not None:
        lines.extend(_summary(view, names, folded, collected))
    return "\n".join(lines) + "\n"


def _hole_lines(view: SeatView, names: Sequence[str]) -> list[str]:
    out = ["*** HOLE CARDS ***"]
    seats = (view.seat,) if view.seat is not None else (0, 1)
    shown = {e["seat"] for e in view.events if e["type"] == "showdown"}
    for s in seats:
        cards = view.holes[s]
        if cards is not None and (s == view.seat or s not in shown):
            out.append(f"Dealt to {names[s]} [{format_cards(cards)}]")
    return out


def _board_before(view: SeatView, street: str) -> list[str]:
    n = {"flop": 0, "turn": 3, "river": 4}[street]
    return [card_to_str(c) for c in view.board[:n]]


def _summary(
    view: SeatView, names: Sequence[str], folded: dict[int, str], collected: list[int]
) -> list[str]:
    result = view.result
    assert result is not None
    lines = ["*** SUMMARY ***", f"Total pot {result.pot} | Rake 0"]
    if view.board:
        lines.append(f"Board [{format_cards(view.board)}]")
    for s in (0, 1):
        tags = " (button) (small blind)" if s == view.button else " (big blind)"
        head = f"Seat {s + 1}: {names[s]}{tags}"
        if s in folded:
            cards = view.holes[s]
            mucked = f" [{format_cards(cards)}]" if cards is not None and view.seat != s else ""
            lines.append(f"{head} folded {_FOLDED_ON[folded[s]]}{mucked}")
        elif result.showdown:
            hand = result.hands[s]
            cards_text = " ".join(hand["cards"])
            if collected[s]:
                lines.append(
                    f"{head} showed [{cards_text}] and won ({collected[s]}) with {hand['rank']}"
                )
            else:
                lines.append(f"{head} showed [{cards_text}] and lost with {hand['rank']}")
        else:
            lines.append(f"{head} collected ({collected[s]})")
    return lines


__all__ = ["format_hand_history"]
