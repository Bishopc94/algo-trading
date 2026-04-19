"""Smoke test the styled HTML notifier.

Mocks the SMTP path and verifies each public notify_* function:
  - Builds a MIMEMultipart with text + html parts
  - HTML contains the expected labels and values
  - Subject line is informative
  - trail_too_tight footer appears when stop_quality says so

Does NOT actually send email. Captures subjects + bodies into a list.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, SRC)

from ai_trade.monitoring import notifier  # noqa: E402

CAPTURE: list[dict] = []


def fake_send(subject: str, html_body: str, text_body: str) -> None:
    CAPTURE.append({"subject": subject, "html": html_body, "text": text_body})


def main() -> int:
    # Redirect the real sender
    notifier._send_async = fake_send  # type: ignore[assignment]

    print("== Test 1: high-conviction signal ==")
    notifier.notify_high_conviction_signal(
        symbol="TEST", strategy="momentum", conviction=0.92,
        hold_type="day", entry_price=10.00, stop_loss=9.70,
        take_profit=10.90, direction="long",
    )
    cap = CAPTURE[-1]
    assert "TEST" in cap["subject"] and "92%" in cap["subject"]
    assert "TEST" in cap["html"]
    assert "$10.00" in cap["html"]
    assert "momentum" in cap["html"]
    assert "1:3.0" in cap["html"]
    print(f"  subject: {cap['subject']}")

    print("\n== Test 2: stock order submitted ==")
    notifier.notify_stock_order(
        symbol="IONZ", strategy="vwap", shares=14, entry_price=8.55,
        stop_loss=8.37, take_profit=9.19, hold_type="day", conviction=0.97,
        order_id="abc-123", cost=119.70,
    )
    cap = CAPTURE[-1]
    assert "IONZ" in cap["subject"] and "14sh" in cap["subject"]
    assert "abc-123" in cap["html"]
    assert "0.97" in cap["html"]
    print(f"  subject: {cap['subject']}")

    print("\n== Test 3: order failed ==")
    notifier.notify_stock_order_failed(
        symbol="BAD", strategy="orb", shares=10, reason="PDT rule",
    )
    cap = CAPTURE[-1]
    assert "FAIL" in cap["subject"]
    assert "PDT rule" in cap["html"]
    print(f"  subject: {cap['subject']}")

    print("\n== Test 4: options order ==")
    notifier.notify_options_order(
        underlying="AAPL", strategy="credit_put_spread", legs=2,
        max_loss=75.00, max_profit=25.00, roi=0.33, conviction=0.80,
        expiration="2026-05-16", order_id="opt-456",
    )
    cap = CAPTURE[-1]
    assert "AAPL" in cap["subject"]
    assert "opt-456" in cap["html"]
    print(f"  subject: {cap['subject']}")

    print("\n== Test 5: trailing stop update (locked-in profit) ==")
    notifier.notify_trailing_stop_update(
        symbol="TSLA", strategy="momentum", old_stop=240.00, new_stop=250.00,
        entry_price=245.00, current_price=260.00, high_since_entry=262.00,
        mode="chandelier", conviction=0.88, atr=3.50, take_profit=280.00,
    )
    cap = CAPTURE[-1]
    assert "TRAIL" in cap["subject"]
    assert "240.00" in cap["subject"] and "250.00" in cap["subject"]
    assert "chandelier" in cap["html"]
    assert "$260.00" in cap["html"]
    assert "profit territory" in cap["html"]  # locked_gain > 0 footer
    print(f"  subject: {cap['subject']}")

    print("\n== Test 6: trailing stop update (still below entry) ==")
    notifier.notify_trailing_stop_update(
        symbol="BELO", strategy="vwap", old_stop=8.00, new_stop=8.30,
        entry_price=8.50, current_price=8.60, high_since_entry=8.65,
        mode="breakeven", conviction=0.72, atr=0.15, take_profit=9.00,
    )
    cap = CAPTURE[-1]
    assert "reduced but not eliminated" in cap["html"]
    print(f"  subject: {cap['subject']}")

    print("\n== Test 7: trade exit -- winner ==")
    notifier.notify_trade_exit(
        symbol="WINZ", strategy="momentum", exit_reason="take_profit",
        entry_price=10.00, exit_price=10.90, shares=20,
        pnl=18.00, pnl_pct=0.09, hold_type="swing", conviction=0.85,
        stop_quality="not_hit", high_since_entry=10.95, take_profit=10.90,
    )
    cap = CAPTURE[-1]
    assert "WIN" in cap["subject"]
    assert "+9.00%" in cap["subject"] or "9.00%" in cap["subject"]
    assert "$18.00" in cap["html"]
    print(f"  subject: {cap['subject']}")

    print("\n== Test 8: trade exit -- trail_too_tight loss ==")
    notifier.notify_trade_exit(
        symbol="IONZ", strategy="vwap", exit_reason="stop_loss",
        entry_price=8.55, exit_price=8.53, shares=14,
        pnl=0.15, pnl_pct=0.0007, hold_type="day", conviction=0.97,
        stop_quality="trail_too_tight", high_since_entry=8.78, take_profit=9.19,
    )
    cap = CAPTURE[-1]
    assert "WIN" in cap["subject"]  # still a slight win, ratchet barely above BE
    assert "TRAIL TOO TIGHT" in cap["html"]
    assert "trail_too_tight" in cap["html"]
    print(f"  subject: {cap['subject']}")

    print(f"\n{len(CAPTURE)} emails captured")
    # Dump first HTML to a file for visual inspection
    out = os.path.join(HERE, "_notifier_sample.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(CAPTURE[4]["html"])  # trailing stop sample
    print(f"Sample HTML written to: {out}")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
