# bot_multi.py — Multi-symbol DCA bot for Bybit (H1-only)
# ENV only: BYBIT_API_KEY, BYBIT_API_SECRET
# Timeframe is fixed to 1 hour (H1). All other settings are constants below.
#
# Features:
#   - H1 bar-close trading (drop in-progress bar)
#   - Startup-guard: no first entry if last CLOSED H1 bar has no signal
#   - DCA-guard: confirm last CLOSED H1 bar (or last N bars) via Bybit public API before DCA
#   - Entry-confirm: require last N CLOSED bars share the same signal before opening
#   - Rate limiting + per-symbol scan spreading (avoid 10006)
#   - Dynamic per-leg notional from equity (eq/120 * leverage), refreshed hourly
#
# Dependencies: pybit==2.*, requests
#
import os
import time
import logging
import zlib
from typing import List, Dict, Optional

from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError

_last_closed_sig_cache = {}


# =====================
# ===== SETTINGS  =====
# =====================
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# Fixed timeframe: H1 only
BARFRAME_SEC     = 3600     # 1 hour
BYBIT_INTERVAL   = "60"     # Bybit kline interval for H1
TF_LABEL         = "1h"

# Symbols file
PAIRS_FILE            = "46cap.txt"

# Sizing / leverage
LEVERAGE_X            = 5.0
LEG_USDT_FALLBACK     = 15.0      # used if equity lookup fails at startup
MIN_NOTIONAL_USDT     = 5.0
VOL_SCALE_LONG        = 1.0
VOL_SCALE_SHORT       = 1.0
MAX_DCA               = -1        # -1 = unlimited

# Fees & funding
TAKER_FEE             = 0.0
FUNDING_RATE_8H       = 0.0

# TP/SL (disabled by default)
TP_PCT                = 0.0
TP_TTL_SEC            = 8.0
TP_POST_ONLY          = True
EMERGENCY_SL_PCT      = 0.0
FLIP_ON_PROFIT        = True

# Guards
ENTRY_CONFIRM_BARS       = 1      # 1 = default; 2 = require last 2 H1 bars have same sig for entry
DCA_REQUIRE_SIGNAL_GUARD = True   # confirm the last CLOSED H1 bar (or seq) via public API
DCA_CONFIRM_BARS         = 1      # 1 = last bar; 2 = last 2 bars same dir
DCA_MIN_DRAWDOWN_PCT     = 0.0    # e.g. 0.015 = 1.5% drawdown required
DCA_GUARD_FAIL_OPEN      = False  # API error → block DCA

# Scheduler / pacing
CLOSE_GRACE_SEC          = 5.0
CLOSE_PASS_RETRIES       = 0
CLOSE_PASS_RETRY_GAP     = 1.5
IDLE_POLL_SEC            = 10.0
STARTUP_PASS             = 1

# Rate limit pacing
RL_SWITCH_SEC            = 0.30
RL_LEV_SEC               = 0.30
RL_KLINE_SEC             = 0.50   # slower for many symbols
RL_MISC_SEC              = 0.12

# Per-symbol scan spread (avoid burst at hh:00)
SCAN_SPREAD_SEC          = 0.0   # spread first requests into this window
START_JITTER_MAX_SEC     = 0.0    # one-time jitter at pass start

# Kline payload
KLINE_LIMIT              = 61     # enough lookback for the signal

# Logging
LOG_LEVEL                = "INFO" # DEBUG/INFO/WARNING/ERROR
LOG_BAR_SIG              = False  # log signal on each closed bar

# =====================
# ====== RUNTIME  =====
# =====================
CATEGORY = "linear"

_level_map = {
    "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
    "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL
}
logging.basicConfig(level=_level_map.get(LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger().setLevel(_level_map.get(LOG_LEVEL.upper(), logging.INFO))

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise SystemExit("BYBIT_API_KEY / BYBIT_API_SECRET are required but missing.")

# ---------- Utils ----------
class RateLimiter:
    def __init__(self, min_interval_sec: float):
        self.min = float(min_interval_sec)
        self.last = 0.0
    def wait(self):
        now = time.monotonic()
        dt = now - self.last
        if dt < self.min:
            time.sleep(self.min - dt)
        self.last = time.monotonic()

rl_switch = RateLimiter(RL_SWITCH_SEC)
rl_lev    = RateLimiter(RL_LEV_SEC)
rl_kline  = RateLimiter(RL_KLINE_SEC)
rl_misc   = RateLimiter(RL_MISC_SEC)

def read_pairs(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip().upper()
            if not s:
                continue
            if not s.endswith("USDT"):
                s += "USDT"
            out.append(s)
    return out

def round_step_floor(x: float, step: float) -> float:
    if step <= 0:
        return x
    k = int(x / step + 1e-12)  # floor
    v = k * step
    return v if v > 0 else step

# Spread helper: deterministic offset per symbol within [0, SCAN_SPREAD_SEC)
def _spread_offset(symbol: str) -> float:
    try:
        h = zlib.adler32(symbol.encode('utf-8')) & 0xffffffff
        frac = (h % 1000000) / 1000000.0
        return frac * float(SCAN_SPREAD_SEC)
    except Exception:
        return 0.0

# --- Lelec-like signal (from 18.py, bar-close) ---
def bulls_signal_from_klines_barclose(klines: List[List[float]]) -> List[int]:
    length = 50
    bars = 30
    o = [float(x[1]) for x in klines]
    h = [float(x[2]) for x in klines]
    l = [float(x[3]) for x in klines]
    c = [float(x[4]) for x in klines]
    n = len(c)
    if n < max(length, 35):
        return [0] * n

    highest = [0.0] * n
    lowest  = [0.0] * n
    for i in range(n):
        lo = max(0, i - length + 1)
        highest[i] = max(h[lo:i+1])
        lowest[i]  = min(l[lo:i+1])

    bindex = 0
    sindex = 0
    lelex  = [0] * n

    for i in range(n):
        if i >= 4 and c[i] > c[i - 4]:
            bindex += 1
        if i >= 4 and c[i] < c[i - 4]:
            sindex += 1

        condShort = (bindex > bars) and (c[i] < o[i]) and (h[i] >= highest[i])
        condLong  = (sindex > bars) and (c[i] > o[i]) and (l[i] <= lowest[i])

        if condShort:
            bindex = 0
            lelex[i] = -1
        elif condLong:
            sindex = 0
            lelex[i] = 1

    return lelex

# ---------- Trader ----------
class Trader:
    def __init__(self, client: HTTP, symbol: str, inst_cfg: Dict[str, float], ctx: "MultiBot"):
        self.client = client
        self.symbol = symbol
        self.disabled = False
        self.ctx = ctx
        self.qty_step = inst_cfg.get("qty_step", 0.001)
        self.min_qty  = inst_cfg.get("min_qty", 0.001)
        self.tick     = inst_cfg.get("tick", 0.001)

        self.dir = 0         # +1/-1/0
        self.qty = 0.0
        self.avg: Optional[float] = None
        self.legs = 0

        self.entry_fees = 0.0
        self.funding_paid = 0.0
        self.funding_anchor = None
        self.last_bar_ts = None
        self.last_close: Optional[float] = None  # cache close

        # TP maker-first state
        self.tp_pending = False
        self.tp_px: Optional[float] = None
        self.tp_deadline = 0.0

    # ---- exchange setup ----
    def _switch_one_way(self):
        try:
            rl_switch.wait()
            self.client.switch_position_mode(category=CATEGORY, symbol=self.symbol, mode=0)
        except InvalidRequestError as e:
            if "110025" not in str(e):
                logging.warning("[%s] switch mode failed: %s", self.symbol, e)
        except Exception as e:
            logging.warning("[%s] switch mode failed: %s", self.symbol, e)

    def _set_leverage(self):
        try:
            rl_lev.wait()
            self.client.set_leverage(category=CATEGORY, symbol=self.symbol,
                                     buyLeverage=str(LEVERAGE_X), sellLeverage=str(LEVERAGE_X))
        except InvalidRequestError as e:
            msg = str(e)
            if "110074" in msg:     # closed contract
                logging.debug("[%s] closed contract; disabling", self.symbol)
                setattr(self, "_disabled", True)
            elif "110043" not in msg:     # leverage not modified → ignore
                logging.warning("[%s] set leverage failed: %s", self.symbol, e)
        except Exception as e:
            logging.warning("[%s] set leverage failed: %s", self.symbol, e)

    # ---- data fetch ----
    def klines_tf(self, limit: int = KLINE_LIMIT) -> List[List[float]]:
        backoff = 0.8
        for attempt in range(6):
            try:
                rl_kline.wait()
                r = self.client.get_kline(category=CATEGORY, symbol=self.symbol, interval=BYBIT_INTERVAL, limit=limit)
                out: List[List[float]] = []
                for it in r["result"]["list"]:
                    ts = int(it[0]); o = float(it[1]); h = float(it[2]); l = float(it[3]); c = float(it[4])
                    out.append([ts, o, h, l, c])
                out.sort(key=lambda x: x[0])
                return out
            except Exception as e:
                msg = str(e).lower()
                if "10006" in msg or "rate limit" in msg:
                    time.sleep(backoff); backoff = min(backoff * 1.6, 8.0); continue
                if attempt < 2:
                    time.sleep(backoff); backoff = min(backoff * 1.6, 8.0); continue
                raise
        raise RuntimeError(f"[{self.symbol}] get_kline failed after retries")

    def _format_qty_for_exchange(self, qty: float) -> str:
        from decimal import Decimal
        step = Decimal(str(self.qty_step))
        q = (Decimal(str(qty)) // step) * step  # floor to step
        if q <= 0:
            q = step
        min_q = Decimal(str(self.min_qty))
        if q < min_q:
            q = min_q
        return format(q.normalize(), 'f')

    def _upnl_from_price(self, px: float) -> float:
        if self.dir == 0 or self.qty <= 0 or self.avg is None:
            return 0.0
        gross = (px / self.avg - 1.0) if self.dir > 0 else (1.0 - px / self.avg)
        px_pnl = self.qty * self.avg * gross
        exit_fee = self.qty * px * TAKER_FEE
        return px_pnl - (self.entry_fees + exit_fee + self.funding_paid)

    # ---- order helpers ----
    def place_market(self, side: str, qty: float, reduce_only: bool = False) -> bool:
        if qty <= 0:
            return False
        try:
            rl_misc.wait()
            self.client.place_order(category=CATEGORY, symbol=self.symbol, side=side,
                                    orderType="Market", qty=self._format_qty_for_exchange(qty),
                                    reduceOnly=reduce_only, timeInForce="IOC")
            return True
        except Exception as e:
            msg = str(e).lower()
            if "symbol is not supported" in msg or ("instrument" in msg and "not" in msg and "support" in msg) or "errcode: 10001" in msg:
                logging.error("[%s] exchange rejected symbol as unsupported; disabling further trading.", self.symbol)
                self.disabled = True
                return False
            logging.warning("[%s] place_market failed: %s", self.symbol, e)
            return False

    def cancel_all_orders(self):
        try:
            rl_misc.wait()
            self.client.cancel_all_orders(category=CATEGORY, symbol=self.symbol)
        except Exception as e:
            logging.debug("[%s] cancel_all_orders failed: %s", self.symbol, e)

    def place_limit(self, side: str, qty: float, price: float, reduce_only: bool = False, post_only: bool = False):
        if qty <= 0:
            return None
        tif = "PostOnly" if post_only else "GoodTillCancel"
        try:
            rl_misc.wait()
            r = self.client.place_order(category=CATEGORY, symbol=self.symbol, side=side,
                                        orderType="Limit", qty=self._format_qty_for_exchange(qty), price=str(price),
                                        reduceOnly=reduce_only, timeInForce=tif)
            return (r.get("result", {}) or {}).get("orderId")
        except Exception as e:
            msg = str(e).lower()
            if "symbol is not supported" in msg or ("instrument" in msg and "not" in msg and "support" in msg) or "errcode: 10001" in msg:
                logging.error("[%s] exchange rejected symbol as unsupported; disabling further trading.", self.symbol)
                self.disabled = True
                return None
            logging.warning("[%s] place_limit failed: %s", self.symbol, e)
            return None

    # ---- position ops ----
    def _fetch_live_price(self) -> float:
        rl_misc.wait()
        r = self.client.get_tickers(category=CATEGORY, symbol=self.symbol)
        return float(r["result"]["list"][0]["lastPrice"])

    def _calc_leg_qty(self, price: float, dir_sign: int) -> float:
        from math import ceil
        idx = self.legs
        scale = VOL_SCALE_LONG if dir_sign > 0 else VOL_SCALE_SHORT
        base_usdt = getattr(self.ctx, 'dynamic_leg_usdt', float(LEG_USDT_FALLBACK)) or float(LEG_USDT_FALLBACK)
        leg_usdt = base_usdt * (scale ** idx)

        qty = leg_usdt / price
        step = self.qty_step
        qty = int(qty / step + 1e-12) * step  # floor to step
        if qty < self.min_qty:
            qty = self.min_qty
        if qty * price < MIN_NOTIONAL_USDT:
            target_qty = ceil(MIN_NOTIONAL_USDT / price / step) * step
            qty = max(qty, target_qty)
        return max(qty, self.min_qty)

    def open_leg(self, dir_sign: int):
        if getattr(self, 'disabled', False):
            return
        price = self._fetch_live_price()
        qty = self._calc_leg_qty(price, dir_sign)
        if qty <= 0:
            logging.info("[%s] skip open leg (qty=0 due to budget/min_qty)", self.symbol)
            return
        side = "Buy" if dir_sign > 0 else "Sell"
        ok = self.place_market(side, qty, reduce_only=False)
        if not ok:
            return
        if self.qty <= 0:
            self.dir = dir_sign
            self.avg = price
            self.qty = qty
            self.legs = 1
        else:
            new_qty = self.qty + qty
            self.avg = (self.avg * self.qty + price * qty) / new_qty
            self.qty = new_qty
            self.legs += 1

        self.entry_fees += qty * price * TAKER_FEE
        if self.funding_anchor is None:
            self.funding_anchor = time.time()

        self.tp_pending = False
        self.tp_px = None
        self.tp_deadline = 0.0

        logging.info("[%s] OPEN leg #%d dir=%+d qty=%.6f @ %.6f (avg=%.6f)",
                     self.symbol, self.legs, self.dir, qty, price, self.avg)

    def close_all(self):
        if self.dir == 0 or self.qty <= 0:
            return
        side = "Sell" if self.dir > 0 else "Buy"
        ok = self.place_market(side, self.qty, reduce_only=True)
        if ok:
            logging.info("[%s] CLOSE ALL qty=%.6f (avg=%.6f)", self.symbol, self.qty, self.avg)
        self.dir = 0
        self.qty = 0.0
        self.avg = None
        self.legs = 0
        self.entry_fees = 0.0
        self.funding_paid = 0.0
        self.funding_anchor = None
        self.tp_pending = False
        self.tp_px = None
        self.tp_deadline = 0.0

    # ---- periodic logic ----
    def _funding_tick(self, price: float):
        if FUNDING_RATE_8H <= 0.0 or self.dir == 0 or self.qty <= 0 or self.funding_anchor is None:
            return
        period = 8.0 * 3600.0
        now = time.time()
        while self.funding_anchor + period <= now:
            notional = self.qty * price
            self.funding_paid += notional * FUNDING_RATE_8H
            self.funding_anchor += period

    def step(self):
        if getattr(self, 'disabled', False):
            return
        kl = self.klines_tf(limit=KLINE_LIMIT)
        if not kl or len(kl) < 61:
            logging.debug("[%s] insufficient klines; skip", self.symbol)
            return
        # === BEGIN: closed-bar & ordering guard ===
        try:
            if len(kl) >= 2 and int(kl[0][0]) > int(kl[-1][0]):
                kl = sorted(kl, key=lambda x: int(x[0]))
        except Exception:
            pass
        try:
            BARFRAME_MS = int(BARFRAME_SEC * 1000)
            _now_ms = int(time.time() * 1000)
            while kl and int(kl[-1][0]) + BARFRAME_MS + GUARD_BUFFER_MS > _now_ms:
                kl = kl[:-1]
        except Exception:
            pass
        # === END: closed-bar & ordering guard ===

        
        # === Enforce strictly the LAST CLOSED bar (sync with remote guard) ===
        try:
            last_open = __bulls__last_open_ms(int(BARFRAME_SEC))
            if self.symbol in DEBUG_ON_SYMBOLS:
                logging.info("[DEBUG][%s] local last_open=%d (ms)", self.symbol, int(last_open))
            kl = [row for row in kl if int(row[0]) <= int(last_open)]
        except Exception:
            pass
        # === END enforce last closed bar ===
        kl_closed = kl
        last_ts = int(kl_closed[-1][0])
        if self.last_bar_ts == last_ts:
            return
        self.last_bar_ts = last_ts
        self.last_close = float(kl_closed[-1][4])

        sig_arr = bulls_signal_from_klines_barclose(kl_closed)
        sig = sig_arr[-1]


        # If local sig==0 but remote gsig!=0, log audit (helps detect nến lệch)
        try:
            if self.dir == 0 and sig == 0:
                _r_sig, _r_last_open = _bulls_get_last_closed_sig_for_symbol(self.symbol)
                if _r_sig != 0:
                    try:
                        _local_last_open = __bulls__last_open_ms(int(BARFRAME_SEC))
                    except Exception:
                        _local_last_open = -1
                    try:
                        _local_last_closed_ts = int(kl_closed[-1][0])
                        _local_close = float(kl_closed[-1][4])
                    except Exception:
                        _local_last_closed_ts = -1
                        _local_close = float('nan')
                    logging.info("[ENTRY-GUARD][SKIP][%s] remote_last_open=%d remote_sig=%+d local_last_open=%d local_last_closed_ts=%d local_sig=%+d local_close=%.6f (local==0)",
                                 self.symbol, int(_r_last_open), int(_r_sig), int(_local_last_open), int(_local_last_closed_ts), int(sig), _local_close)
