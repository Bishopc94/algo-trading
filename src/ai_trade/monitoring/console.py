"""Pretty console output formatting for the trading bot.

Provides consistent, human-readable console output with visual hierarchy,
timestamps, and clear section separators. All bot console output should
go through these helpers for a unified look.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# ── Box-drawing characters ────────────────────────────────
# Using Unicode box-drawing for clean visual structure.
_TOP_LEFT = "╔"
_TOP_RIGHT = "╗"
_BOT_LEFT = "╚"
_BOT_RIGHT = "╝"
_HORIZ = "═"
_VERT = "║"
_THIN_HORIZ = "─"
_SECTION_WIDTH = 60


def _ts() -> str:
    """Current time in ET as [HH:MM AM/PM]."""
    return datetime.now(ET).strftime("%I:%M %p")


def _pad(text: str, width: int = _SECTION_WIDTH) -> str:
    """Pad text to fixed width for box alignment."""
    return text.ljust(width)[:width]


# ══════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════


def banner(version: str, mode: str, equity: float, cash: float,
           stock_strats: int, options_strats: int, pdt_remaining: int) -> str:
    """Startup banner with account info."""
    w = _SECTION_WIDTH
    lines = [
        f"{_TOP_LEFT}{_HORIZ * w}{_TOP_RIGHT}",
        f"{_VERT}{_pad(f'  AI Trade Bot v{version}  [{mode}]', w)}{_VERT}",
        f"{_VERT}{_pad(f'  Equity: ${equity:,.2f}  |  Cash: ${cash:,.2f}', w)}{_VERT}",
        f"{_VERT}{_pad(f'  Strategies: {stock_strats} stock, {options_strats} options  |  PDT: {pdt_remaining} remaining', w)}{_VERT}",
        f"{_BOT_LEFT}{_HORIZ * w}{_BOT_RIGHT}",
        "  Press Ctrl+C to stop.\n",
    ]
    return "\n" + "\n".join(lines)


def section(title: str) -> str:
    """Section header like: [09:45 AM] ── Pre-market Scan ──────────"""
    ts = _ts()
    label = f"[{ts}] {_THIN_HORIZ}{_THIN_HORIZ} {title} "
    remaining = max(0, _SECTION_WIDTH + 4 - len(label))
    return f"\n{label}{_THIN_HORIZ * remaining}"


def info(msg: str) -> str:
    """Indented info line."""
    return f"  {msg}"


def detail(msg: str) -> str:
    """Double-indented detail line."""
    return f"    {msg}"


def success(msg: str) -> str:
    """Success indicator."""
    return f"  + {msg}"


def warning(msg: str) -> str:
    """Warning indicator."""
    return f"  ! {msg}"


def error(msg: str) -> str:
    """Error indicator."""
    return f"  X {msg}"


def skip(msg: str) -> str:
    """Skipped/filtered item."""
    return f"  - {msg}"


def order_submitted(symbol: str, shares: int, entry: float, cost: float,
                    hold_type: str, strategy: str, stop: float,
                    target: float, pdt_remaining: int | None = None) -> str:
    """Format a successful order submission."""
    rr = (target - entry) / max(entry - stop, 0.01)
    lines = [
        f"  >>> ORDER {symbol}  {shares} shares @ ${entry:.2f} (${cost:,.2f})",
        f"      Strategy: {strategy}  |  Hold: {hold_type.upper()}",
        f"      Stop: ${stop:.2f}  |  Target: ${target:.2f}  |  R:R 1:{rr:.1f}",
    ]
    if pdt_remaining is not None:
        lines.append(f"      PDT slots remaining: {pdt_remaining}")
    return "\n".join(lines)


def order_failed(symbol: str, strategy: str, shares: int) -> str:
    """Format a failed order."""
    return f"  X ORDER FAILED  {symbol}  |  {strategy}  |  {shares} shares"


def signal_line(symbol: str, strategy: str, conviction: float,
                hold_type: str, entry: float, stop: float,
                target: float) -> str:
    """Format a signal in the execution queue."""
    rr = (target - entry) / max(entry - stop, 0.01)
    return (
        f"    SIGNAL  {symbol:<6} {strategy:<18} "
        f"conv={conviction:.2f}  {hold_type:<5}  "
        f"entry=${entry:.2f}  stop=${stop:.2f}  target=${target:.2f}  "
        f"R:R 1:{rr:.1f}"
    )


def options_signal_line(underlying: str, strategy: str, roi: float,
                        conviction: float, max_loss: float,
                        max_profit: float) -> str:
    """Format an options signal."""
    return (
        f"    SIGNAL  {underlying:<6} {strategy:<22} "
        f"ROI={roi:.1f}x  conv={conviction:.2f}  "
        f"risk=${max_loss:.2f}  reward=${max_profit:.2f}"
    )


def options_order(underlying: str, strategy: str, legs: int,
                  max_loss: float, max_profit: float, roi: float,
                  expiration: str) -> str:
    """Format a successful options order."""
    return (
        f"  >>> OPTIONS  {underlying}  |  {strategy}  |  {legs} leg(s)\n"
        f"      Risk: ${max_loss:.2f}  |  Reward: ${max_profit:.2f}  "
        f"|  ROI: {roi:.1f}x  |  Exp: {expiration}"
    )


def price_adapted(symbol: str, signal_entry: float, current: float,
                  old_stop: float, new_stop: float,
                  old_target: float, new_target: float,
                  old_shares: int, new_shares: int) -> str:
    """Format bracket order price adaptation."""
    return (
        f"    Price adapted: {symbol}  signal@${signal_entry:.2f} -> now@${current:.2f}\n"
        f"      Stop: ${old_stop:.2f} -> ${new_stop:.2f}  |  "
        f"Target: ${old_target:.2f} -> ${new_target:.2f}  |  "
        f"Shares: {old_shares} -> {new_shares}  "
        f"(${old_shares * signal_entry:,.0f} -> ${new_shares * current:,.0f})"
    )


def regime_line(regime: str, spy_rsi: float, spy_trend: str,
                vix_level: float, vix_trend: str) -> str:
    """Format market regime info."""
    return (
        f"  Regime: {regime.upper()}  |  SPY RSI={spy_rsi:.1f} ({spy_trend})"
        f"  |  VIX={vix_level:.1f} ({vix_trend})"
    )


def regime_modifiers(conviction_mod: float, size_mod: float,
                     longs_allowed: bool, options_biases: str) -> str:
    """Format regime modifiers."""
    longs = "allowed" if longs_allowed else "BLOCKED"
    return (
        f"  Modifiers: conviction={conviction_mod}x  |  size={size_mod}x"
        f"  |  longs={longs}  |  options: [{options_biases}]"
    )


def scan_result(total: int, momentum: int, mean_rev: int, vwap: int,
                symbols: list[str]) -> str:
    """Format scan results."""
    top = ", ".join(symbols[:8])
    more = f" +{len(symbols) - 8} more" if len(symbols) > 8 else ""
    return (
        f"  Found {total} candidates  "
        f"(momentum={momentum}, mean_rev={mean_rev}, vwap={vwap})\n"
        f"  Top: {top}{more}"
    )


def daily_summary(today: str, equity: float, cash: float,
                  open_positions: int, day_trades_used: int,
                  metrics: dict) -> str:
    """Format end-of-day summary box."""
    w = 50
    sep = _THIN_HORIZ * w

    pnl = metrics["total_pnl"]
    pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"

    lines = [
        f"\n{sep}",
        f"  Daily Summary {_THIN_HORIZ} {today}",
        f"{sep}",
        f"  Equity:  ${equity:,.2f}     Cash:  ${cash:,.2f}",
        f"  Open:    {open_positions} positions    PDT:   {day_trades_used}/3 used",
        f"{sep}",
        f"  Trades closed:  {metrics['total_trades']}",
        f"  Win rate:       {metrics['win_rate']:.0%}",
        f"  Total P&L:      {pnl_str}",
        f"  Avg win:        ${metrics['avg_win']:,.2f}     Avg loss:  ${metrics['avg_loss']:,.2f}",
        f"  Profit factor:  {metrics['profit_factor']}",
        f"  Sharpe ratio:   {metrics['sharpe_ratio']}",
        f"  Max drawdown:   {metrics['max_drawdown_pct']:.1f}%",
        f"{sep}",
    ]
    return "\n".join(lines)


def cycle_summary(
    regime: str, vix: float, pdt_used: int, pdt_max: int,
    candidates: int, momentum: int, mean_rev: int, vwap: int,
    signals: list[dict], near_misses: list[dict],
    equity: float, cash: float, open_positions: int, heat_pct: float,
) -> str:
    """V2: Rich per-cycle summary matching the V2 agent brief spec.

    Args:
        signals: list of {symbol, strategy, conviction, entry, stop, target, hold_type, action}
        near_misses: list of {symbol, strategy, reason, miss_pct}
    """
    ts = _ts()
    w = _SECTION_WIDTH + 4
    sep = _HORIZ * w

    lines = [f"\n{sep}", f"  Scan {ts}  |  Regime: {regime.upper()}  |  VIX {vix:.1f}  |  PDT: {pdt_used}/{pdt_max}"]
    lines.append(f"  Scanned: {candidates} candidates (momentum={momentum}, mean_rev={mean_rev}, vwap={vwap})")

    if signals:
        lines.append("")
        lines.append("  Signals:")
        for i, s in enumerate(signals, 1):
            risk = s["entry"] - s["stop"]
            reward = s["target"] - s["entry"]
            rr = reward / risk if risk > 0 else 0
            action_str = s.get("action", "QUEUED").upper()
            shares_str = f"({s['shares']} shares, ${s['shares'] * s['entry']:,.0f})" if s.get("shares") else ""
            lines.append(
                f"    {i}. {s['symbol']:<6} {s['strategy']:<16} "
                f"conv={s['conviction']:.2f}  entry=${s['entry']:.2f}  "
                f"stop=${s['stop']:.2f}  target=${s['target']:.2f}  "
                f"R:R=1:{rr:.1f}  -> {action_str} {shares_str}"
            )
    else:
        lines.append("")
        lines.append("  No signals this cycle.")

    if near_misses:
        lines.append("")
        lines.append("  Near-misses:")
        for nm in near_misses[:5]:
            miss_str = f" -- missed by {nm['miss_pct']:.1f}%" if nm.get("miss_pct") else ""
            lines.append(f"    - {nm['symbol']:<6} {nm['strategy']}: {nm['reason']}{miss_str}")

    lines.append("")
    lines.append(f"  Portfolio: ${equity:,.2f} equity | ${cash:,.2f} cash | {open_positions} open | heat {heat_pct:.1f}%")
    lines.append(sep)

    return "\n".join(lines)


def catchup(msg: str) -> str:
    """Catchup status message."""
    return f"  [Catchup] {msg}"


def stopped() -> str:
    """Bot stopped message."""
    return "\n  Bot stopped.\n"
