"""Email notification system for trade alerts.

Sends email alerts when:
  - A signal with conviction >= 0.70 is generated
  - A stock or options order is submitted (or fails)

Uses Gmail SMTP with an app password stored in the .env file.
All emails are sent in a background thread to avoid blocking the
trading pipeline.
"""

from __future__ import annotations

import os
import smtplib
import threading
from datetime import datetime
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

from ai_trade.monitoring.logger import get_logger

logger = get_logger(__name__)

ET = ZoneInfo("America/New_York")

# Loaded once from environment — set in config/.env
_SMTP_USER: str = ""
_SMTP_PASS: str = ""
_RECIPIENT: str = "bishopchristopher94@gmail.com"
_SMTP_HOST: str = "smtp.gmail.com"
_SMTP_PORT: int = 587


def _load_creds() -> tuple[str, str]:
    """Return (user, password) from env, caching after first call."""
    global _SMTP_USER, _SMTP_PASS  # noqa: PLW0603
    if not _SMTP_USER:
        _SMTP_USER = os.environ.get("SMTP_USER", "")
        _SMTP_PASS = os.environ.get("SMTP_PASS", "")
    return _SMTP_USER, _SMTP_PASS


def _send_email(subject: str, body: str) -> None:
    """Send an email via Gmail SMTP. Runs in a background thread."""
    user, password = _load_creds()
    if not user or not password:
        logger.debug("email_skipped_no_creds")
        return

    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = _RECIPIENT

    try:
        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        logger.debug("email_sent", subject=subject)
    except Exception:
        logger.exception("email_send_failed", subject=subject)


def _send_async(subject: str, body: str) -> None:
    """Fire-and-forget email in a daemon thread."""
    t = threading.Thread(target=_send_email, args=(subject, body), daemon=True)
    t.start()


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
    now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    subject = f"HIGH CONVICTION {conviction:.0%} — {symbol} ({strategy})"
    body = (
        f"High-conviction signal detected\n"
        f"{'=' * 40}\n"
        f"Time:        {now}\n"
        f"Symbol:      {symbol}\n"
        f"Strategy:    {strategy}\n"
        f"Direction:   {direction}\n"
        f"Hold type:   {hold_type}\n"
        f"Conviction:  {conviction:.2f}\n"
        f"\n"
        f"Entry:       ${entry_price:.2f}\n"
        f"Stop loss:   ${stop_loss:.2f}\n"
        f"Take profit: ${take_profit:.2f}\n"
        f"Risk/reward: 1:{(take_profit - entry_price) / max(entry_price - stop_loss, 0.01):.1f}\n"
    )
    _send_async(subject, body)


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
    now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    subject = f"ORDER SUBMITTED — {symbol} {shares} shares @ ${entry_price:.2f}"
    body = (
        f"Stock order submitted\n"
        f"{'=' * 40}\n"
        f"Time:        {now}\n"
        f"Symbol:      {symbol}\n"
        f"Strategy:    {strategy}\n"
        f"Hold type:   {hold_type}\n"
        f"Conviction:  {conviction:.2f}\n"
        f"\n"
        f"Shares:      {shares}\n"
        f"Entry:       ${entry_price:.2f}\n"
        f"Stop loss:   ${stop_loss:.2f}\n"
        f"Take profit: ${take_profit:.2f}\n"
        f"Total cost:  ${cost:.2f}\n"
        f"Risk/reward: 1:{(take_profit - entry_price) / max(entry_price - stop_loss, 0.01):.1f}\n"
        f"\n"
        f"Order ID:    {order_id}\n"
    )
    _send_async(subject, body)


def notify_stock_order_failed(
    symbol: str,
    strategy: str,
    shares: int,
) -> None:
    """Send an alert when a stock order fails."""
    now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    subject = f"ORDER FAILED — {symbol} ({strategy})"
    body = (
        f"Stock order FAILED\n"
        f"{'=' * 40}\n"
        f"Time:    {now}\n"
        f"Symbol:  {symbol}\n"
        f"Strategy: {strategy}\n"
        f"Shares:  {shares}\n"
    )
    _send_async(subject, body)


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
    now = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    subject = f"OPTIONS ORDER — {underlying} ({strategy}) ROI={roi:.1f}x"
    body = (
        f"Options order submitted\n"
        f"{'=' * 40}\n"
        f"Time:        {now}\n"
        f"Underlying:  {underlying}\n"
        f"Strategy:    {strategy}\n"
        f"Conviction:  {conviction:.2f}\n"
        f"\n"
        f"Legs:        {legs}\n"
        f"Max loss:    ${max_loss:.2f}\n"
        f"Max profit:  ${max_profit:.2f}\n"
        f"ROI:         {roi:.1f}x\n"
        f"Expiration:  {expiration}\n"
        f"\n"
        f"Order ID:    {order_id}\n"
    )
    _send_async(subject, body)
