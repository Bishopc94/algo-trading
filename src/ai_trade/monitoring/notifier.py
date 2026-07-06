"""Email notification system for trade alerts.

Sends styled HTML emails for key trading events:
  - High-conviction signal generated
  - Stock or options order submitted (or failed)
  - Trailing stop tightened on an open position
  - Trade closed (exit)

Uses Gmail SMTP with an app password stored in the .env file.
All emails are sent in a background thread to avoid blocking the
trading pipeline.
"""

from __future__ import annotations

import html
import os
import smtplib
import threading
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

from ai_trade.monitoring.logger import get_logger

logger = get_logger(__name__)

ET = ZoneInfo("America/New_York")

# Loaded once from environment -- set in config/.env
_SMTP_USER: str = ""
_SMTP_PASS: str = ""
_RECIPIENT: str = "bishopchristopher94@gmail.com"
_SMTP_HOST: str = "smtp.gmail.com"
_SMTP_PORT: int = 587


# ── Palette ───────────────────────────────────────────────────────────────
# Muted, readable colors that render consistently in Gmail/Outlook/mobile.

_COLORS = {
    "bg": "#0f1419",
    "card": "#1a1f2e",
    "border": "#2a3142",
    "text": "#e8eaed",
    "muted": "#9aa0a6",
    "accent": "#4fc3f7",
    "green": "#4caf50",
    "red": "#ef5350",
    "amber": "#ffb74d",
    "blue": "#42a5f5",
}


def _load_creds() -> tuple[str, str]:
    """Return (user, password) from env, caching after first call."""
    global _SMTP_USER, _SMTP_PASS  # noqa: PLW0603
    if not _SMTP_USER:
        _SMTP_USER = os.environ.get("SMTP_USER", "")
        _SMTP_PASS = os.environ.get("SMTP_PASS", "")
    return _SMTP_USER, _SMTP_PASS


def _send_email(subject: str, html_body: str, text_body: str) -> None:
    """Send a multipart email via Gmail SMTP. Runs in a background thread."""
    user, password = _load_creds()
    if not user or not password:
        logger.debug("email_skipped_no_creds", subject=subject)
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = _RECIPIENT
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        logger.debug("email_sent", subject=subject)
    except Exception:
        logger.exception("email_send_failed", subject=subject)


def _send_async(subject: str, html_body: str, text_body: str) -> None:
    """Fire-and-forget email in a daemon thread."""
    t = threading.Thread(
        target=_send_email, args=(subject, html_body, text_body), daemon=True,
    )
    t.start()


def _now_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")


# ── HTML building blocks ──────────────────────────────────────────────────


def _render_html(
    title: str,
    subtitle: str,
    accent: str,
    rows: list[tuple[str, str]],
    footer: str | None = None,
) -> str:
    """Render a card-style HTML email.

    ``rows`` is a list of (label, value) pairs. Values are escaped.
    ``accent`` is a hex color for the left-edge bar.
    """
    c = _COLORS
    row_html = "".join(
        f"""
        <tr>
          <td style="padding:8px 14px;color:{c['muted']};font-size:13px;
                     text-align:left;width:42%;
                     border-bottom:1px solid {c['border']};">{html.escape(label)}</td>
          <td style="padding:8px 14px;color:{c['text']};font-size:14px;
                     font-weight:600;text-align:right;
                     border-bottom:1px solid {c['border']};
                     font-family:Menlo,Consolas,monospace;">
            {html.escape(value)}
          </td>
        </tr>
        """
        for label, value in rows
    )
    foot = ""
    if footer:
        foot = f"""
        <div style="padding:14px 18px;color:{c['muted']};font-size:12px;
                    border-top:1px solid {c['border']};background:{c['bg']};
                    line-height:1.5;">
          {html.escape(footer)}
        </div>
        """

    return f"""<!doctype html>
<html><body style="margin:0;padding:24px;background:{c['bg']};
                   font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  <table role="presentation" cellpadding="0" cellspacing="0" border="0"
         style="max-width:560px;margin:0 auto;background:{c['card']};
                border-radius:10px;overflow:hidden;
                border-left:5px solid {accent};
                border:1px solid {c['border']};
                border-left:5px solid {accent};">
    <tr><td style="padding:20px 20px 10px 20px;">
      <div style="color:{accent};font-size:12px;font-weight:700;
                  letter-spacing:1.4px;text-transform:uppercase;">
        {html.escape(subtitle)}
      </div>
      <div style="color:{c['text']};font-size:22px;font-weight:700;
                  margin-top:4px;">
        {html.escape(title)}
      </div>
    </td></tr>
    <tr><td style="padding:4px 6px 12px 6px;">
      <table role="presentation" cellpadding="0" cellspacing="0" border="0"
             width="100%" style="border-collapse:collapse;">
        {row_html}
      </table>
    </td></tr>
    {foot}
  </table>
</body></html>
"""


def _render_text(title: str, rows: list[tuple[str, str]], footer: str | None = None) -> str:
    lines = [title, "=" * 48]
    width = max((len(k) for k, _ in rows), default=0) + 2
    for label, value in rows:
        lines.append(f"{label:<{width}}{value}")
    if footer:
        lines.append("")
        lines.append(footer)
    return "\n".join(lines) + "\n"


def _fmt_price(x: float | None) -> str:
    if x is None:
        return "--"
    return f"${x:.2f}"


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "--"
    return f"{x * 100:+.2f}%"


def _rr(entry: float, stop: float, target: float) -> str:
    risk = entry - stop
    reward = target - entry
    if risk <= 0:
        return "--"
    return f"1:{reward / risk:.1f}"


# ── Public API ────────────────────────────────────────────────


def notify_high_conviction_signal(
    symbol: str,
    strategy: str,
    conviction: float,
    hold_type: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: str = "long",
) -> None:
    """Send an alert when a signal has conviction >= 0.70."""
    subject = f"[SIGNAL] {symbol} {conviction:.0%} conviction ({strategy})"
    rows = [
        ("Time", _now_et()),
        ("Symbol", symbol),
        ("Strategy", strategy),
        ("Direction", direction),
        ("Hold type", hold_type),
        ("Conviction", f"{conviction:.2f}  ({conviction:.0%})"),
        ("Entry", _fmt_price(entry_price)),
        ("Stop loss", _fmt_price(stop_loss)),
        ("Take profit", _fmt_price(take_profit)),
        ("Risk:Reward", _rr(entry_price, stop_loss, take_profit)),
    ]
    accent = _COLORS["amber"] if conviction < 0.85 else _COLORS["green"]
    html_body = _render_html(
        title=f"{symbol} -- high conviction signal",
        subtitle="Signal generated",
        accent=accent,
        rows=rows,
        footer="No order placed yet. This fires at the signal stage, before risk sizing.",
    )
    text_body = _render_text(f"HIGH CONVICTION SIGNAL -- {symbol}", rows)
    _send_async(subject, html_body, text_body)


def notify_stock_order(
    symbol: str,
    strategy: str,
    shares: int,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    hold_type: str,
    conviction: float,
    order_id: str,
    cost: float,
) -> None:
    """Send an alert when a stock bracket order is submitted."""
    subject = f"[ORDER] {symbol} {shares}sh @ {_fmt_price(entry_price)}"
    risk_amount = shares * max(0.0, entry_price - stop_loss)
    reward_amount = shares * max(0.0, take_profit - entry_price)
    rows = [
        ("Time", _now_et()),
        ("Symbol", symbol),
        ("Strategy", strategy),
        ("Hold type", hold_type),
        ("Conviction", f"{conviction:.2f}  ({conviction:.0%})"),
        ("Shares", f"{shares}"),
        ("Entry", _fmt_price(entry_price)),
        ("Stop loss", _fmt_price(stop_loss)),
        ("Take profit", _fmt_price(take_profit)),
        ("Risk:Reward", _rr(entry_price, stop_loss, take_profit)),
        ("Risk $", _fmt_price(risk_amount)),
        ("Reward $", _fmt_price(reward_amount)),
        ("Total cost", _fmt_price(cost)),
        ("Order ID", order_id),
    ]
    html_body = _render_html(
        title=f"{symbol} -- order submitted",
        subtitle="Bracket order placed",
        accent=_COLORS["blue"],
        rows=rows,
    )
    text_body = _render_text(f"ORDER SUBMITTED -- {symbol}", rows)
    _send_async(subject, html_body, text_body)


def notify_stock_order_failed(
    symbol: str,
    strategy: str,
    shares: int,
    reason: str = "",
) -> None:
    """Send an alert when a stock order fails."""
    subject = f"[FAIL] {symbol} order rejected ({strategy})"
    rows = [
        ("Time", _now_et()),
        ("Symbol", symbol),
        ("Strategy", strategy),
        ("Shares", f"{shares}"),
    ]
    if reason:
        rows.append(("Reason", reason))
    html_body = _render_html(
        title=f"{symbol} -- order failed",
        subtitle="Bracket order rejected",
        accent=_COLORS["red"],
        rows=rows,
        footer="Check the console log and Alpaca account status.",
    )
    text_body = _render_text(f"ORDER FAILED -- {symbol}", rows)
    _send_async(subject, html_body, text_body)


def notify_options_order(
    underlying: str,
    strategy: str,
    legs: int,
    max_loss: float,
    max_profit: float,
    roi: float,
    conviction: float,
    expiration: str,
    order_id: str,
) -> None:
    """Send an alert when an options order is submitted."""
    subject = f"[OPTIONS] {underlying} {strategy} ROI={roi:.1f}x"
    rows = [
        ("Time", _now_et()),
        ("Underlying", underlying),
        ("Strategy", strategy),
        ("Conviction", f"{conviction:.2f}  ({conviction:.0%})"),
        ("Legs", f"{legs}"),
        ("Max loss", _fmt_price(max_loss)),
        ("Max profit", _fmt_price(max_profit)),
        ("ROI", f"{roi:.1f}x"),
        ("Expiration", expiration),
        ("Order ID", order_id),
    ]
    html_body = _render_html(
        title=f"{underlying} -- options order",
        subtitle="Options order placed",
        accent=_COLORS["accent"],
        rows=rows,
    )
    text_body = _render_text(f"OPTIONS ORDER -- {underlying}", rows)
    _send_async(subject, html_body, text_body)


def notify_trailing_stop_update(
    symbol: str,
    strategy: str,
    old_stop: float,
    new_stop: float,
    entry_price: float,
    current_price: float,
    high_since_entry: float | None,
    mode: str,
    conviction: float,
    atr: float | None,
    take_profit: float | None,
) -> None:
    """Send an alert when a trailing stop tightens on an open position."""
    unrealized = current_price - entry_price
    unrealized_pct = unrealized / entry_price if entry_price > 0 else 0.0
    locked_gain = new_stop - entry_price
    locked_pct = locked_gain / entry_price if entry_price > 0 else 0.0

    subject = (
        f"[TRAIL] {symbol} stop {old_stop:.2f} -> {new_stop:.2f} ({mode})"
    )
    rows = [
        ("Time", _now_et()),
        ("Symbol", symbol),
        ("Strategy", strategy),
        ("Trail mode", mode),
        ("Conviction", f"{conviction:.2f}  ({conviction:.0%})"),
        ("Entry", _fmt_price(entry_price)),
        ("Current price", _fmt_price(current_price)),
        ("Unrealized", f"{_fmt_price(unrealized)}  ({_fmt_pct(unrealized_pct)})"),
        ("Peak (since entry)", _fmt_price(high_since_entry)),
        ("Target", _fmt_price(take_profit)),
        ("ATR", _fmt_price(atr)),
        ("Old stop", _fmt_price(old_stop)),
        ("New stop", _fmt_price(new_stop)),
        ("Locked-in", f"{_fmt_price(locked_gain)}  ({_fmt_pct(locked_pct)})"),
    ]
    if locked_gain >= 0:
        accent = _COLORS["green"]
        footer = "Stop now sits in profit territory -- the trade can only close green from here."
    else:
        accent = _COLORS["amber"]
        footer = "Stop tightened but still below entry; downside is reduced but not eliminated."
    html_body = _render_html(
        title=f"{symbol} -- stop tightened",
        subtitle="Trailing stop update",
        accent=accent,
        rows=rows,
        footer=footer,
    )
    text_body = _render_text(f"TRAILING STOP -- {symbol}", rows, footer=footer)
    _send_async(subject, html_body, text_body)


def notify_trade_exit(
    symbol: str,
    strategy: str,
    exit_reason: str,
    entry_price: float,
    exit_price: float,
    shares: int,
    pnl: float,
    pnl_pct: float,
    hold_type: str,
    conviction: float | None,
    stop_quality: str | None,
    high_since_entry: float | None,
    take_profit: float | None,
    low_since_entry: float | None = None,
    hold_hours: float | None = None,
) -> None:
    """Send an alert when a trade closes."""
    won = pnl >= 0
    tag = "WIN" if won else "LOSS"
    subject = (
        f"[{tag}] {symbol} {_fmt_pct(pnl_pct)}  pnl={_fmt_price(pnl)}  ({exit_reason})"
    )

    # Compute MFE/MAE — surfaces "give-back" / "near miss" patterns.
    mfe_pct = mae_pct = None
    capture_pct = None
    if entry_price and entry_price > 0:
        if high_since_entry:
            mfe_pct = (high_since_entry - entry_price) / entry_price * 100
            if mfe_pct > 0:
                capture_pct = pnl_pct / mfe_pct * 100
        if low_since_entry:
            mae_pct = (entry_price - low_since_entry) / entry_price * 100

    rows = [
        ("Time", _now_et()),
        ("Symbol", symbol),
        ("Strategy", strategy),
        ("Hold type", hold_type),
        ("Exit reason", exit_reason),
    ]
    if hold_hours is not None:
        if hold_hours >= 24:
            rows.append(("Held", f"{hold_hours/24:.1f} days"))
        else:
            rows.append(("Held", f"{hold_hours:.1f} hours"))
    if conviction is not None:
        rows.append(("Conviction", f"{conviction:.2f}  ({conviction:.0%})"))
    rows.extend([
        ("Shares", f"{shares}"),
        ("Entry", _fmt_price(entry_price)),
        ("Exit", _fmt_price(exit_price)),
        ("Peak (MFE)", f"{_fmt_price(high_since_entry)}" +
                       (f"  ({_fmt_pct(mfe_pct)})" if mfe_pct is not None else "")),
        ("Trough (MAE)", f"{_fmt_price(low_since_entry)}" +
                         (f"  ({_fmt_pct(-mae_pct)})" if mae_pct is not None else "")),
        ("Target", _fmt_price(take_profit)),
        ("P&L", f"{_fmt_price(pnl)}  ({_fmt_pct(pnl_pct)})"),
    ])
    if capture_pct is not None and mfe_pct is not None and mfe_pct >= 3.0:
        # Useful only when there WAS meaningful upside to capture
        rows.append(("MFE capture", f"{capture_pct:.0f}% of peak"))
    if stop_quality:
        rows.append(("Stop quality", stop_quality))

    footer = None
    if stop_quality == "trail_too_tight":
        footer = (
            "Stop quality flagged as TRAIL TOO TIGHT: the trailing logic "
            "ratcheted the stop and killed a profitable trade before it "
            "could reach target. Review trailing parameters for this "
            "strategy."
        )
    elif stop_quality == "too_tight":
        footer = "Price reversed past the stop and recovered -- the stop was too tight for the volatility."
    elif stop_quality == "too_loose":
        footer = "Heavy slippage past stop before fill -- the stop level was too far from a real invalidation point."

    accent = _COLORS["green"] if won else _COLORS["red"]
    html_body = _render_html(
        title=f"{symbol} -- {tag.lower()} closed",
        subtitle=f"Trade exit ({exit_reason})",
        accent=accent,
        rows=rows,
        footer=footer,
    )
    text_body = _render_text(
        f"TRADE CLOSED -- {symbol} ({tag})", rows, footer=footer,
    )
    _send_async(subject, html_body, text_body)


def notify_bot_downtime(gap_hours: float, last_heartbeat_utc: str) -> None:
    """Email the operator that the bot was down (silent crash or stop).

    Triggered at startup when the persisted heartbeat is older than the
    configured threshold.  Surfaces three things the operator cares about
    after a gap: how long the bot was offline, when it was last alive,
    and a reminder that server-side brackets on Alpaca were the only
    protection during the gap (so any positions held overnight rode
    their original stop/target — see RGNT 2026-06-10).
    """
    if gap_hours >= 24:
        gap_human = f"{gap_hours/24:.1f} days"
    else:
        gap_human = f"{gap_hours:.1f} hours"
    subject = f"[BOT DOWN] Bot was offline for {gap_human}"
    rows = [
        ("Restart time", _now_et()),
        ("Gap", gap_human),
        ("Last heartbeat (UTC)", last_heartbeat_utc),
    ]
    footer = (
        "Trailing-stop adjustments and exit logic did NOT run during the "
        "gap. Open positions were protected only by their server-side "
        "brackets on Alpaca, so any stops or targets that hit during the "
        "gap filled at the available market price (potentially with "
        "slippage on a gap-down). Review trades that closed in the window "
        "to see if any need attention."
    )
    accent = _COLORS["red"]
    html_body = _render_html(
        title="Bot was offline",
        subtitle=f"Downtime alert -- {gap_human}",
        accent=accent,
        rows=rows,
        footer=footer,
    )
    text_body = _render_text("BOT DOWNTIME ALERT", rows, footer=footer)
    _send_async(subject, html_body, text_body)


def notify_ml_promotion_blocked(
    version: int,
    val_accuracy: float,
    training_trades: int,
    prior_active: dict | None,
    reasons: list[str],
) -> None:
    """Email the operator that the latest model failed the promotion gate.

    The new version is still saved on disk and registered as inactive so
    we can inspect it after the fact, but the prior active model keeps
    serving inference.  The 2026-06 incident — v5 at 12.5% acc auto-
    promoted over v3 at 60.9% — is exactly the scenario this prevents.
    """
    rows = [
        ("New version", f"v{version}"),
        ("Val accuracy", f"{val_accuracy:.3f}"),
        ("Training trades", str(training_trades)),
    ]
    if prior_active:
        prior_acc = prior_active.get("backtest_accuracy")
        rows.append((
            "Prior active",
            f"v{prior_active.get('version')} -- acc "
            f"{(float(prior_acc) if prior_acc is not None else 0.0):.3f}, "
            f"trades {prior_active.get('training_trades')}",
        ))
    rows.append(("Failed checks", "; ".join(reasons) if reasons else "n/a"))
    subject = f"[ML] Promotion blocked -- v{version} failed quality gate"
    footer = (
        "The new model was saved on disk and recorded with is_active=0 "
        "for later inspection. Live inference continues against the prior "
        "active model. Investigate the training pipeline (label quality, "
        "feature drift, sample size) before forcing promotion."
    )
    accent = _COLORS["red"]
    html_body = _render_html(
        title="ML promotion blocked",
        subtitle=f"v{version} failed the quality gate",
        accent=accent,
        rows=rows,
        footer=footer,
    )
    text_body = _render_text("ML PROMOTION BLOCKED", rows, footer=footer)
    _send_async(subject, html_body, text_body)


def notify_ml_rollback(
    failed_version: int,
    failed_val_accuracy: float | None,
    failed_training_trades: int,
    restored_version: int | None,
    restored_val_accuracy: float | None,
    restored_training_trades: int | None,
    reasons: list[str],
) -> None:
    """Email the operator that a broken active model was auto-rolled-back.

    Counterpart to the trainer's promotion gate: the trainer blocks a bad
    model from going active, this fires when the predictor finds an
    already-active model that fails the same gate at startup and demotes
    it in favour of the most recent passing version.  The 2026-06 incident
    — v6 at 12.5% acc hand-promoted over v3 at 60.9% — is the scenario
    this self-heals on the next restart.
    """
    rows = [
        ("Demoted version", f"v{failed_version}"),
        (
            "Demoted val accuracy",
            f"{failed_val_accuracy:.3f}" if failed_val_accuracy is not None else "n/a",
        ),
        ("Demoted training trades", str(failed_training_trades)),
    ]
    if restored_version is not None:
        rows.append((
            "Restored active",
            f"v{restored_version} -- acc "
            f"{(restored_val_accuracy if restored_val_accuracy is not None else 0.0):.3f}, "
            f"trades {restored_training_trades}",
        ))
    else:
        rows.append(("Restored active", "none -- cold-start (no passing model)"))
    rows.append(("Failed checks", "; ".join(reasons) if reasons else "n/a"))
    subject = (
        f"[ML] Auto-rollback -- v{failed_version} demoted"
        + (f", v{restored_version} restored" if restored_version is not None else "")
    )
    footer = (
        "The active model failed the quality gate at startup and was "
        "demoted to is_active=0. The most recent passing version was "
        "marked active in its place; if none passed, inference falls back "
        "to rule-based conviction (cold-start). No retrain was triggered."
    )
    accent = _COLORS["red"]
    html_body = _render_html(
        title="ML auto-rollback",
        subtitle=f"v{failed_version} demoted at startup",
        accent=accent,
        rows=rows,
        footer=footer,
    )
    text_body = _render_text("ML AUTO-ROLLBACK", rows, footer=footer)
    _send_async(subject, html_body, text_body)
