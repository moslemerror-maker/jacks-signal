from __future__ import annotations

SLIPPAGE = 0.0005


def charges(entry: float, exit_: float, qty: int, side: str, slippage: float = SLIPPAGE) -> float:
    buy, sell = (entry, exit_) if side == "LONG" else (exit_, entry)
    bv, sv = buy * qty, sell * qty
    brokerage = min(20.0, 0.0003 * bv) + min(20.0, 0.0003 * sv)
    stt = 0.00025 * sv
    exch = 0.0000307 * (bv + sv)
    sebi = 0.000001 * (bv + sv)
    stamp = 0.00003 * bv
    gst = 0.18 * (brokerage + exch + sebi)
    return brokerage + stt + exch + sebi + stamp + gst + slippage * (bv + sv)
