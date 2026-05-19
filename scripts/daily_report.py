#!/usr/bin/env python3
"""Generate a daily TradingAgents watchlist triage report.

The report combines:
- the latest saved TradingAgents debate rating per symbol
- fresh daily yfinance price/volume data
- simple trigger rules for names that deserve a fresh full debate

It is designed for the weekday 06:45 Europe/Berlin automation. At that time the
US session is closed and the report is useful before gettex/Xetra trading.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import yfinance as yf

from scripts.batch_debate import (
    DEFAULT_SYMBOL_MAP,
    RATING_SCORE,
    WatchlistRow,
    extract_rating,
    read_watchlist,
)


@dataclass
class DailyRow:
    company: str
    trade_symbol: str
    analysis_symbol: str
    baseline_rating: str
    baseline_date: str
    latest_price_date: str
    close: float | None
    pct_1d: float | None
    pct_5d: float | None
    pct_20d: float | None
    rsi_14: float | None
    volume_ratio_20d: float | None
    vs_sma_50: float | None
    vs_sma_200: float | None
    triggers: list[str]
    priority: int
    log_path: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--watchlist",
        default="watchlists/ai_infra_watchlist.csv",
        help="Watchlist CSV path.",
    )
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--timezone", default="Europe/Berlin")
    parser.add_argument("--period", default="1y", help="yfinance download period")
    parser.add_argument(
        "--results-roots",
        default="./results,./results_debate,./results_debate_5",
        help="Comma-separated directories containing TradingAgents logs.",
    )
    parser.add_argument("--output-dir", default="./results/daily_reports")
    parser.add_argument("--max-action-board", type=int, default=12)
    parser.add_argument("--big-1d-move", type=float, default=3.0)
    parser.add_argument("--big-5d-move", type=float, default=7.0)
    parser.add_argument("--volume-spike", type=float, default=1.8)
    parser.add_argument("--stale-days", type=int, default=4)
    return parser.parse_args()


def latest_log_for_symbol(
    symbol: str,
    report_date: str,
    roots: list[Path],
) -> tuple[Path | None, str]:
    best: tuple[str, Path] | None = None
    for root in roots:
        log_dir = root / symbol / "TradingAgentsStrategy_logs"
        if not log_dir.exists():
            continue
        for path in log_dir.glob("full_states_log_*.json"):
            log_date = path.stem.replace("full_states_log_", "")
            if log_date <= report_date and (best is None or log_date > best[0]):
                best = (log_date, path)
    if best is None:
        return None, ""
    return best[1], best[0]


def baseline_rating(symbol: str, report_date: str, roots: list[Path]) -> tuple[str, str, str]:
    path, log_date = latest_log_for_symbol(symbol, report_date, roots)
    if not path:
        return "UNKNOWN", "", ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rating = extract_rating(payload.get("final_trade_decision", ""))
    return rating, log_date, str(path)


def clean_price_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[-1] if col[0] == "Price" else col[0] for col in data.columns]
    data = data.copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["Close"])


def pct_change(close: pd.Series, periods: int) -> float | None:
    if len(close) <= periods:
        return None
    base = close.iloc[-periods - 1]
    latest = close.iloc[-1]
    if not base or pd.isna(base) or pd.isna(latest):
        return None
    return (latest / base - 1) * 100


def rsi(close: pd.Series, window: int = 14) -> float | None:
    if len(close) <= window:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    latest_loss = loss.iloc[-1]
    if pd.isna(latest_loss):
        return None
    if latest_loss == 0:
        return 100.0
    value = 100 - (100 / (1 + gain.iloc[-1] / latest_loss))
    return float(value) if not pd.isna(value) else None


def pct_vs_sma(close: pd.Series, window: int) -> float | None:
    if len(close) < window:
        return None
    sma = close.rolling(window).mean().iloc[-1]
    if pd.isna(sma) or sma == 0:
        return None
    return (close.iloc[-1] / sma - 1) * 100


def fmt_num(value: float | None, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "n/a"
    return f"{value:.2f}{suffix}"


def analyze_price(
    row: WatchlistRow,
    args: argparse.Namespace,
    report_dt: date,
) -> tuple[dict[str, float | str | None], list[str], int, str]:
    try:
        data = yf.download(
            row.analysis_symbol,
            period=args.period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        data = clean_price_frame(data)
        if data.empty:
            return {}, ["NO PRICE DATA"], 90, "No yfinance rows returned"

        close = data["Close"]
        volume = data["Volume"] if "Volume" in data else pd.Series(dtype=float)
        latest_date = data.index[-1].date()
        latest_close = float(close.iloc[-1])
        pct_1d = pct_change(close, 1)
        pct_5d = pct_change(close, 5)
        pct_20d = pct_change(close, 20)
        rsi_14 = rsi(close, 14)
        volume_ratio = None
        if len(volume) >= 21 and volume.iloc[-20:-1].mean() > 0:
            volume_ratio = float(volume.iloc[-1] / volume.iloc[-20:-1].mean())

        metrics = {
            "latest_price_date": latest_date.isoformat(),
            "close": latest_close,
            "pct_1d": pct_1d,
            "pct_5d": pct_5d,
            "pct_20d": pct_20d,
            "rsi_14": rsi_14,
            "volume_ratio_20d": volume_ratio,
            "vs_sma_50": pct_vs_sma(close, 50),
            "vs_sma_200": pct_vs_sma(close, 200),
        }

        triggers = []
        priority = 0
        age_days = (report_dt - latest_date).days
        if age_days > args.stale_days:
            triggers.append(f"STALE PRICE {age_days}d")
            priority += 50
        if pct_1d is not None and abs(pct_1d) >= args.big_1d_move:
            triggers.append(f"1D MOVE {pct_1d:+.1f}%")
            priority += 20
        if pct_5d is not None and abs(pct_5d) >= args.big_5d_move:
            triggers.append(f"5D MOVE {pct_5d:+.1f}%")
            priority += 18
        if volume_ratio is not None and volume_ratio >= args.volume_spike:
            triggers.append(f"VOLUME {volume_ratio:.1f}x")
            priority += 14
        if rsi_14 is not None and rsi_14 >= 70:
            triggers.append(f"RSI HIGH {rsi_14:.0f}")
            priority += 8
        if rsi_14 is not None and rsi_14 <= 30:
            triggers.append(f"RSI LOW {rsi_14:.0f}")
            priority += 8

        return metrics, triggers, priority, ""
    except Exception as exc:
        return {}, ["PRICE ERROR"], 80, str(exc)


def build_daily_rows(args: argparse.Namespace) -> list[DailyRow]:
    symbol_map = dict(DEFAULT_SYMBOL_MAP)
    watchlist = read_watchlist(args.watchlist, symbol_map)
    roots = [Path(root.strip()) for root in args.results_roots.split(",") if root.strip()]
    report_dt = datetime.strptime(args.date, "%Y-%m-%d").date()
    rows = []

    for item in watchlist:
        rating, rating_date, log_path = baseline_rating(item.analysis_symbol, args.date, roots)
        metrics, triggers, price_priority, error = analyze_price(item, args, report_dt)

        if rating in {"SELL", "UNDERWEIGHT"}:
            price_priority += 4
        elif rating == "HOLD":
            price_priority += 2
        elif rating == "UNKNOWN":
            price_priority += 10

        rows.append(
            DailyRow(
                company=item.notes,
                trade_symbol=item.trade_symbol,
                analysis_symbol=item.analysis_symbol,
                baseline_rating=rating,
                baseline_date=rating_date,
                latest_price_date=str(metrics.get("latest_price_date", "")),
                close=metrics.get("close"),
                pct_1d=metrics.get("pct_1d"),
                pct_5d=metrics.get("pct_5d"),
                pct_20d=metrics.get("pct_20d"),
                rsi_14=metrics.get("rsi_14"),
                volume_ratio_20d=metrics.get("volume_ratio_20d"),
                vs_sma_50=metrics.get("vs_sma_50"),
                vs_sma_200=metrics.get("vs_sma_200"),
                triggers=triggers,
                priority=price_priority + max(0, 2 - RATING_SCORE.get(rating, 0)),
                log_path=log_path,
                error=error,
            )
        )
    return rows


def write_markdown(rows: list[DailyRow], args: argparse.Namespace) -> Path:
    tz = ZoneInfo(args.timezone)
    now = datetime.now(tz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{args.date}_0645_daily_triage.md"

    action_rows = sorted(rows, key=lambda r: (-r.priority, r.company))[: args.max_action_board]
    buy_count = sum(1 for r in rows if r.baseline_rating == "BUY")
    hold_count = sum(1 for r in rows if r.baseline_rating == "HOLD")
    sell_count = sum(1 for r in rows if r.baseline_rating == "SELL")
    triggered = [r for r in rows if r.triggers]

    lines = [
        f"# TradingAgents Daily Triage - {args.date}",
        "",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "## Snapshot",
        "",
        f"- Universe: {len(rows)} symbols",
        f"- Baseline debate ratings: {buy_count} BUY, {hold_count} HOLD, {sell_count} SELL",
        f"- Triggered for attention: {len(triggered)}",
        "- Data source: yfinance daily bars; use broker quotes before execution.",
        "",
        "## Action Board",
        "",
        "| Company | Symbol | Baseline | Close | 1D | 5D | RSI | Vol | Trigger |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in action_rows:
        lines.append(
            "| {company} | {symbol} | {rating} | {close} | {pct_1d} | {pct_5d} | {rsi} | {vol} | {triggers} |".format(
                company=row.company,
                symbol=row.analysis_symbol,
                rating=row.baseline_rating,
                close=fmt_num(row.close),
                pct_1d=fmt_num(row.pct_1d, "%"),
                pct_5d=fmt_num(row.pct_5d, "%"),
                rsi=fmt_num(row.rsi_14),
                vol=fmt_num(row.volume_ratio_20d, "x"),
                triggers=", ".join(row.triggers) if row.triggers else "Watch",
            )
        )

    lines += [
        "",
        "## Fresh Debate Candidates",
        "",
        "| Company | Symbol | Baseline | Reason | Latest Bar |",
        "|---|---:|---:|---|---:|",
    ]
    for row in sorted(triggered, key=lambda r: (-r.priority, r.company)):
        lines.append(
            "| {company} | {symbol} | {rating} | {reason} | {bar_date} |".format(
                company=row.company,
                symbol=row.analysis_symbol,
                rating=row.baseline_rating,
                reason=", ".join(row.triggers),
                bar_date=row.latest_price_date or "n/a",
            )
        )
    if not triggered:
        lines.append("| None | n/a | n/a | No configured trigger fired | n/a |")

    lines += [
        "",
        "## Full Universe Appendix",
        "",
        "| Company | Symbol | Rating | Close | 1D | 5D | 20D | vs 50D | vs 200D | Price Date |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (row.baseline_rating, row.company)):
        lines.append(
            "| {company} | {symbol} | {rating} | {close} | {pct_1d} | {pct_5d} | {pct_20d} | {sma50} | {sma200} | {price_date} |".format(
                company=row.company,
                symbol=row.analysis_symbol,
                rating=row.baseline_rating,
                close=fmt_num(row.close),
                pct_1d=fmt_num(row.pct_1d, "%"),
                pct_5d=fmt_num(row.pct_5d, "%"),
                pct_20d=fmt_num(row.pct_20d, "%"),
                sma50=fmt_num(row.vs_sma_50, "%"),
                sma200=fmt_num(row.vs_sma_200, "%"),
                price_date=row.latest_price_date or "n/a",
            )
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = report_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "company",
                "trade_symbol",
                "analysis_symbol",
                "baseline_rating",
                "baseline_date",
                "latest_price_date",
                "close",
                "pct_1d",
                "pct_5d",
                "pct_20d",
                "rsi_14",
                "volume_ratio_20d",
                "vs_sma_50",
                "vs_sma_200",
                "triggers",
                "priority",
                "log_path",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            payload = row.__dict__.copy()
            payload["triggers"] = "; ".join(row.triggers)
            writer.writerow(payload)

    candidates_path = out_dir / f"{args.date}_0645_fresh_debate_candidates.csv"
    with candidates_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trade_symbol",
                "underlying_symbol",
                "session_focus",
                "notes",
                "trigger_reason",
            ],
        )
        writer.writeheader()
        for row in sorted(triggered, key=lambda r: (-r.priority, r.company)):
            writer.writerow(
                {
                    "trade_symbol": row.trade_symbol,
                    "underlying_symbol": row.analysis_symbol,
                    "session_focus": "triggered",
                    "notes": row.company,
                    "trigger_reason": "; ".join(row.triggers),
                }
            )

    latest_path = out_dir / "latest_daily_triage.md"
    latest_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_candidates_path = out_dir / "latest_fresh_debate_candidates.csv"
    latest_candidates_path.write_text(candidates_path.read_text(encoding="utf-8"), encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    rows = build_daily_rows(args)
    report_path = write_markdown(rows, args)
    print(f"Wrote {report_path}")
    print(f"Wrote {report_path.with_suffix('.csv')}")
    print(f"Wrote {report_path.parent / (args.date + '_0645_fresh_debate_candidates.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
