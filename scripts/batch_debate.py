#!/usr/bin/env python3
"""Run or summarize TradingAgents debates for a CSV watchlist.

Input CSV columns:
    trade_symbol,underlying_symbol,session_focus,notes

By default this script reuses existing TradingAgents JSON logs and writes a
portfolio summary. Use --run to execute missing or forced analyses.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tradingagents.default_config import DEFAULT_CONFIG


RATING_RE = re.compile(
    r"(?:rating|recommendation|decision|final transaction proposal)\s*[:\-]?\s*"
    r"(?:\*\*)?\b(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)\b",
    re.IGNORECASE,
)

RATING_SCORE = {
    "BUY": 2,
    "OVERWEIGHT": 1,
    "HOLD": 0,
    "UNDERWEIGHT": -1,
    "SELL": -2,
}

# Ambiguous/non-primary tickers from the provided GETTEX/XETR watchlist.
# These keep yfinance pointed at the intended listing instead of a same-symbol
# unrelated security.
DEFAULT_SYMBOL_MAP = {
    "HY9H": "HY9H.MU",  # SK hynix German listing
    "VSA": "VSA.MU",    # Valeo German listing
    "ASX": "AUX.MU",    # ASX Limited German listing; ASX alone is different
    "AUX": "AUX.MU",
    "SRAG": "SRAG.MU",  # Samara Asset Group German listing
    "RHM": "RHM.DE",    # Rheinmetall XETRA
}


@dataclass
class WatchlistRow:
    trade_symbol: str
    underlying_symbol: str
    session_focus: str
    notes: str
    analysis_symbol: str


@dataclass
class DebateSummary:
    trade_symbol: str
    underlying_symbol: str
    analysis_symbol: str
    session_focus: str
    company: str
    analysis_date: str
    rating: str
    action: str
    score: int
    log_path: str
    status: str
    error: str = ""


def read_watchlist(path: str, symbol_map: dict[str, str]) -> list[WatchlistRow]:
    if path == "-":
        text = sys.stdin.read()
        rows = csv.DictReader(text.splitlines())
    else:
        rows = csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines())

    required = {"trade_symbol", "underlying_symbol", "session_focus", "notes"}
    if not rows.fieldnames or not required.issubset(rows.fieldnames):
        missing = sorted(required - set(rows.fieldnames or []))
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

    watchlist = []
    for row in rows:
        underlying = row["underlying_symbol"].strip()
        trade_symbol = row["trade_symbol"].strip()
        analysis_symbol = resolve_analysis_symbol(trade_symbol, underlying, symbol_map)
        watchlist.append(
            WatchlistRow(
                trade_symbol=trade_symbol,
                underlying_symbol=underlying,
                session_focus=row["session_focus"].strip(),
                notes=row["notes"].strip(),
                analysis_symbol=analysis_symbol,
            )
        )
    return watchlist


def resolve_analysis_symbol(
    trade_symbol: str,
    underlying_symbol: str,
    symbol_map: dict[str, str],
) -> str:
    if "." in underlying_symbol:
        return underlying_symbol

    if underlying_symbol in symbol_map:
        return symbol_map[underlying_symbol]

    market, _, code = trade_symbol.partition(":")
    if market.upper() == "XETR":
        return f"{underlying_symbol}.DE"

    return underlying_symbol


def load_symbol_map(path: str | None) -> dict[str, str]:
    symbol_map = dict(DEFAULT_SYMBOL_MAP)
    if not path:
        return symbol_map

    map_path = Path(path)
    if map_path.suffix.lower() == ".json":
        symbol_map.update(json.loads(map_path.read_text(encoding="utf-8")))
        return symbol_map

    with map_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol_map[row["underlying_symbol"].strip()] = row[
                "analysis_symbol"
            ].strip()
    return symbol_map


def find_existing_log(
    symbol: str,
    analysis_date: str,
    results_roots: list[Path],
) -> Path | None:
    rel = Path(symbol) / "TradingAgentsStrategy_logs" / f"full_states_log_{analysis_date}.json"
    for root in results_roots:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def extract_rating(text: str) -> str:
    match = RATING_RE.search(text or "")
    if match:
        return match.group(1).upper()

    upper = (text or "").upper()
    positions = [(upper.find(rating), rating) for rating in RATING_SCORE if rating in upper]
    positions = [(idx, rating) for idx, rating in positions if idx >= 0]
    if positions:
        return min(positions)[1]
    return "UNKNOWN"


def action_from_rating(rating: str) -> str:
    if rating in {"BUY", "OVERWEIGHT"}:
        return "BUY"
    if rating in {"SELL", "UNDERWEIGHT"}:
        return "SELL"
    if rating == "HOLD":
        return "HOLD"
    return "UNKNOWN"


def summary_from_log(row: WatchlistRow, log_path: Path) -> DebateSummary:
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    final_decision = payload.get("final_trade_decision", "")
    rating = extract_rating(final_decision)
    return DebateSummary(
        trade_symbol=row.trade_symbol,
        underlying_symbol=row.underlying_symbol,
        analysis_symbol=row.analysis_symbol,
        session_focus=row.session_focus,
        company=row.notes,
        analysis_date=str(payload.get("trade_date", "")),
        rating=rating,
        action=action_from_rating(rating),
        score=RATING_SCORE.get(rating, 0),
        log_path=str(log_path),
        status="existing",
    )


def run_analysis(row: WatchlistRow, args: argparse.Namespace) -> DebateSummary:
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = DEFAULT_CONFIG.copy()
    config["results_dir"] = args.output_dir
    config["llm_provider"] = args.llm_provider
    config["quick_think_llm"] = args.quick_llm
    config["deep_think_llm"] = args.deep_llm
    config["codex_model_reasoning_effort"] = args.codex_reasoning_effort
    config["max_debate_rounds"] = args.debate_rounds
    config["max_risk_discuss_rounds"] = args.risk_rounds
    config["output_language"] = args.output_language
    config["data_vendors"] = {
        "core_stock_apis": args.stock_vendor,
        "technical_indicators": args.indicator_vendor,
        "fundamental_data": args.fundamental_vendor,
        "news_data": args.news_vendor,
    }

    graph = TradingAgentsGraph(
        selected_analysts=args.analysts.split(","),
        debug=args.debug,
        config=config,
    )
    final_state, processed_rating = graph.propagate(row.analysis_symbol, args.date)
    rating = extract_rating(processed_rating or final_state.get("final_trade_decision", ""))
    log_path = (
        Path(args.output_dir)
        / row.analysis_symbol
        / "TradingAgentsStrategy_logs"
        / f"full_states_log_{args.date}.json"
    )
    return DebateSummary(
        trade_symbol=row.trade_symbol,
        underlying_symbol=row.underlying_symbol,
        analysis_symbol=row.analysis_symbol,
        session_focus=row.session_focus,
        company=row.notes,
        analysis_date=args.date,
        rating=rating,
        action=action_from_rating(rating),
        score=RATING_SCORE.get(rating, 0),
        log_path=str(log_path),
        status="ran",
    )


def write_outputs(summaries: list[DebateSummary], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(s) for s in sorted(summaries, key=lambda s: (-s.score, s.company))]

    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    md_path = output_prefix.with_suffix(".md")
    md_lines = [
        "# Batch Debate Summary",
        "",
        "| Company | Analysis Symbol | Rating | Action | Source |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            "| {company} | {analysis_symbol} | {rating} | {action} | {status} |".format(
                **row
            )
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("watchlist", help="CSV watchlist path, or '-' for stdin")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-dir", default=os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument(
        "--existing-roots",
        default="./results,./results_debate,./results_debate_5,./results_debate_10",
        help="Comma-separated result directories to search before running.",
    )
    parser.add_argument("--symbol-map", default=None, help="CSV or JSON symbol override map")
    parser.add_argument("--run", action="store_true", help="Run analyses that do not have logs")
    parser.add_argument("--force", action="store_true", help="Run even when an existing log is found")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--llm-provider", default=DEFAULT_CONFIG["llm_provider"])
    parser.add_argument("--quick-llm", default=DEFAULT_CONFIG["quick_think_llm"])
    parser.add_argument("--deep-llm", default=DEFAULT_CONFIG["deep_think_llm"])
    parser.add_argument(
        "--codex-reasoning-effort",
        default=DEFAULT_CONFIG.get("codex_model_reasoning_effort"),
        help="Codex model_reasoning_effort config value, e.g. xhigh.",
    )
    parser.add_argument(
        "--debate-rounds",
        type=int,
        default=DEFAULT_CONFIG["max_debate_rounds"],
        help="Full bull/bear investment debate rounds to run.",
    )
    parser.add_argument(
        "--risk-rounds",
        type=int,
        default=DEFAULT_CONFIG["max_risk_discuss_rounds"],
        help="Full aggressive/conservative/neutral risk debate rounds to run.",
    )
    parser.add_argument("--output-language", default="English")
    parser.add_argument("--analysts", default="market,social,news,fundamentals")
    parser.add_argument("--stock-vendor", default="yfinance")
    parser.add_argument("--indicator-vendor", default="yfinance")
    parser.add_argument("--fundamental-vendor", default="yfinance")
    parser.add_argument("--news-vendor", default="yfinance")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbol_map = load_symbol_map(args.symbol_map)
    watchlist = read_watchlist(args.watchlist, symbol_map)
    roots = [Path(root.strip()) for root in args.existing_roots.split(",") if root.strip()]
    summaries = []

    for row in watchlist:
        try:
            existing_log = find_existing_log(row.analysis_symbol, args.date, roots)
            if existing_log and not args.force:
                summary = summary_from_log(row, existing_log)
            elif args.run or args.force:
                summary = run_analysis(row, args)
            else:
                summary = DebateSummary(
                    trade_symbol=row.trade_symbol,
                    underlying_symbol=row.underlying_symbol,
                    analysis_symbol=row.analysis_symbol,
                    session_focus=row.session_focus,
                    company=row.notes,
                    analysis_date=args.date,
                    rating="UNKNOWN",
                    action="UNKNOWN",
                    score=0,
                    log_path="",
                    status="missing",
                    error="No existing log found. Re-run with --run to analyze.",
                )
            summaries.append(summary)
            print(f"{row.analysis_symbol}: {summary.rating} ({summary.status})")
        except Exception as exc:
            summaries.append(
                DebateSummary(
                    trade_symbol=row.trade_symbol,
                    underlying_symbol=row.underlying_symbol,
                    analysis_symbol=row.analysis_symbol,
                    session_focus=row.session_focus,
                    company=row.notes,
                    analysis_date=args.date,
                    rating="UNKNOWN",
                    action="UNKNOWN",
                    score=0,
                    log_path="",
                    status="error",
                    error=str(exc),
                )
            )
            print(f"{row.analysis_symbol}: ERROR {exc}", file=sys.stderr)

    output_prefix = Path(
        args.output_prefix
        or Path(args.output_dir) / f"batch_debate_summary_{args.date}"
    )
    write_outputs(summaries, output_prefix)
    print(f"Wrote {output_prefix.with_suffix('.csv')}")
    print(f"Wrote {output_prefix.with_suffix('.json')}")
    print(f"Wrote {output_prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
