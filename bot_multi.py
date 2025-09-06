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
        except Exception:
            pass
        price_ref = self.last_close

        if LOG_BAR_SIG:
            try:
                ts_sec = last_ts // 1000 if last_ts > 10**12 else last_ts
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts_sec))
            except Exception:
                ts_str = str(last_ts)
            logging.info("[%s] BAR %s | tf=%s | sig=%+d dir=%+d legs=%d upnl=%.4f close=%.6f",
                         self.symbol, ts_str, TF_LABEL, sig, self.dir, self.legs, self._upnl_from_price(price_ref), self.last_close)

        # Flip immediately if single-leg and signal flipped
        if self.dir != 0 and self.legs == 1 and sig == -self.dir:
            try:
                gsig, _ = _bulls_get_last_closed_sig_for_symbol(self.symbol)
                if gsig != -self.dir:
                    logging.info("[%s] FLIP-GUARD(l1): confirm failed gsig=%+d need=%+d → skip flip", self.symbol, gsig, -self.dir)
                else:
                    if getattr(self, "tp_pending", False):
                        try:
                            self.cancel_all_orders()
                        except Exception:
                            pass
                        self.tp_pending = False
                        self.tp_px = None
                        self.tp_deadline = 0.0
                    self.close_all()
                    self.open_leg(-self.dir)
                    return
            except Exception as e:
                logging.warning("[%s] FLIP-GUARD(l1): API err=%s → block flip", self.symbol, e)
                pass

        self._funding_tick(price_ref)

        # TP / SL
        if self.dir != 0 and self.qty > 0 and self.avg is not None:
            if TP_PCT > 0.0:
                want_tp = self.avg * (1.0 + self.dir * TP_PCT)
                if not self.tp_pending:
                    reached = (self.dir > 0 and price_ref >= want_tp) or (self.dir < 0 and price_ref <= want_tp)
                    if reached:
                        tp_px = round_step_floor(want_tp, self.tick)
                        side = "Sell" if self.dir > 0 else "Buy"
                        oid = self.place_limit(side, self.qty, tp_px, reduce_only=True, post_only=TP_POST_ONLY)
                        if oid:
                            self.tp_pending = True
                            self.tp_px = tp_px
                            self.tp_deadline = time.time() + TP_TTL_SEC
                            logging.info("[%s] TP limit placed qty=%.6f @ %.6f (TTL %.1fs)",
                                         self.symbol, self.qty, tp_px, TP_TTL_SEC)
                        else:
                            self.close_all()
                            return
                else:
                    if time.time() >= self.tp_deadline:
                        self.cancel_all_orders()
                        self.close_all()
                        self.tp_pending = False
                        self.tp_px = None
                        self.tp_deadline = 0.0
                        return

            if EMERGENCY_SL_PCT > 0.0:
                sl_px = self.avg * (1.0 - self.dir * EMERGENCY_SL_PCT)
                hit = (self.dir > 0 and price_ref <= sl_px) or (self.dir < 0 and price_ref >= sl_px)
                if hit:
                    if self.tp_pending:
                        self.cancel_all_orders()
                        self.tp_pending = False
                        self.tp_px = None
                        self.tp_deadline = 0.0
                    self.close_all()
                    return

        # ---- ENTRY when flat ----

        if self.dir == 0:

            if sig != 0:

                ok_entry = True

                if ENTRY_CONFIRM_BARS > 1:

                    if len(sig_arr) >= ENTRY_CONFIRM_BARS:

                        tail = sig_arr[-ENTRY_CONFIRM_BARS:]

                        ok_entry = all(s == sig for s in tail)

                    else:

                        ok_entry = False

                if ok_entry:

                    try:

                        gsig, _r_last_open = _bulls_get_last_closed_sig_for_symbol(self.symbol)

                        if gsig != sig:

                            # Comprehensive audit log for ALL symbols when ENTRY-GUARD skips

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

                            logging.info(

                                "[ENTRY-GUARD][SKIP][%s] remote_last_open=%d remote_sig=%+d local_last_open=%d local_last_closed_ts=%d local_sig=%+d local_close=%.6f",

                                self.symbol, int(_r_last_open), int(gsig), int(_local_last_open), int(_local_last_closed_ts), int(sig), _local_close,

                            )

                        else:

                            self.open_leg(sig)

                    except Exception as e:

                        logging.warning("[%s] ENTRY-GUARD: API err=%s → block entry", self.symbol, e)

                        pass

                else:

                    logging.info("[%s] ENTRY-GUARD: require last %d bars sig=%+d; tail=%s → skip entry",

                                 self.symbol, ENTRY_CONFIRM_BARS, sig, str(sig_arr[-ENTRY_CONFIRM_BARS:]))

                return



        # ---- POSITION MANAGEMENT ----
        upnl = self._upnl_from_price(price_ref)

        # DCA / Flip logic
        if sig == self.dir:
            if upnl < 0.0:
                # min drawdown
                dd_pct = 0.0
                if self.avg is not None and self.avg > 0:
                    dd_pct = (price_ref / self.avg - 1.0) if self.dir > 0 else (1.0 - price_ref / self.avg)
                if DCA_MIN_DRAWDOWN_PCT > 0.0 and dd_pct > -DCA_MIN_DRAWDOWN_PCT:
                    logging.info("[%s] DCA-GUARD: skip (drawdown %.3f%% > -%.3f%% req)",
                                 self.symbol, dd_pct*100.0, DCA_MIN_DRAWDOWN_PCT*100.0)
                else:
                    ok_guard = True
                    if DCA_REQUIRE_SIGNAL_GUARD:
                        try:
                            if DCA_CONFIRM_BARS <= 1:
                                gsig, _ = _bulls_get_last_closed_sig_for_symbol(self.symbol)
                                ok_guard = (gsig == self.dir)
                                if not ok_guard:
                                    logging.info("[%s] DCA-GUARD: confirm failed gsig=%+d need=%+d → skip DCA", self.symbol, gsig, self.dir)
                            else:
                                seq = _bulls_get_sig_seq_for_last_n(self.symbol, DCA_CONFIRM_BARS)
                                ok_guard = (len(seq) == DCA_CONFIRM_BARS and all(s == self.dir for s in seq))
                                if not ok_guard:
                                    logging.info("[%s] DCA-GUARD: confirm failed seq=%s need=%+d x%d → skip DCA",
                                                 self.symbol, seq, self.dir, DCA_CONFIRM_BARS)
                        except Exception as e:
                            ok_guard = DCA_GUARD_FAIL_OPEN
                            logging.warning("[%s] DCA-GUARD: API err=%s → %s", self.symbol, e,
                                            "fail-open (allow DCA)" if ok_guard else "block DCA")
                    if ok_guard:
                        if MAX_DCA < 0 or self.legs < 1 + MAX_DCA:
                            self.open_leg(self.dir)
                        else:
                            logging.info("[%s] MAX_DCA reached; skip DCA", self.symbol)
        elif sig == -self.dir:
            if self.legs == 1:
                self.close_all()
                self.open_leg(-self.dir)
                return
            else:
                if upnl >= 0.0 or (FLIP_ON_PROFIT and upnl > 0.0):
                    self.close_all()
                    self.open_leg(-self.dir)
                    return

        # Breakeven exit after multiple legs
        if self.legs > 1 and upnl >= 0.0 and sig == 0:
            self.close_all()

    def init_on_exchange(self):
        if getattr(self, "_disabled", False):
            return
        self._switch_one_way()
        self._set_leverage()

# ---------- Time helpers ----------
def _next_bar_close(now: Optional[float] = None) -> int:
    if now is None:
        now = time.time()
    bf = int(BARFRAME_SEC)
    return (int(now) // bf) * bf + bf

# ---------- MultiBot ----------
class MultiBot:
    def __init__(self, symbols: List[str]):
        self._skipped = []
        self.client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET, recv_window=5000)
        self._err_gate: Dict[str, float] = {}
        self.symbols = self._filter_live(symbols)
        self.traders: Dict[str, Trader] = {}
        inst_map = self._load_instruments()
        for s in self.symbols:
            cfg = inst_map.get(s, {"qty_step": 0.001, "min_qty": 0.001, "tick": 0.001})
            t = Trader(self.client, s, cfg, ctx=self)
            t.init_on_exchange()
            self.traders[s] = t
        self.dynamic_leg_usdt = float(LEG_USDT_FALLBACK)

        if self._skipped:
            preview = ", ".join(self._skipped[:20])
            more = "" if len(self._skipped) <= 20 else f" (+{len(self._skipped)-20} more)"
            logging.info("Skipped %d non-live symbols: %s%s", len(self._skipped), preview, more)
        self._last_budget_hour = -1  # hour marker
        self._startup = True
        self.STARTUP_REQUIRE_SIGNAL = True
        self._startup_skipped_syms = []
        self._startup_opened_syms = []

        logging.info("MultiBot started for %d symbols; TF=%s (%ds) | LEG_USDT=%.4f, lev=%.1fx, fee=%.4f, funding_8h=%.6f | ENTRY_CONFIRM_BARS=%d | DCA_CONFIRM_BARS=%d",
                     len(self.traders), TF_LABEL, int(BARFRAME_SEC), LEG_USDT_FALLBACK, LEVERAGE_X, TAKER_FEE, FUNDING_RATE_8H, ENTRY_CONFIRM_BARS, DCA_CONFIRM_BARS)

        # Startup: compute hourly base leg now (with leverage)
        try:
            eq0 = self._safe_get_equity_usdt()
            if eq0 > 0.0:
                self.dynamic_leg_usdt = max(round((eq0 / 120.0) * LEVERAGE_X, 6), 1e-6)
                logging.info("[BUDGET] (startup) equity=%.6f → dynamic_leg_usdt=%.6f (with leverage %.1fx)", eq0, self.dynamic_leg_usdt, LEVERAGE_X)
        except Exception as e:
            logging.warning("[BUDGET] startup update failed: %s", e)

    def _load_instruments(self) -> Dict[str, Dict[str, float]]:
        m: Dict[str, Dict[str, float]] = {}
        for s in self.symbols:
            try:
                rl_misc.wait()
                r = self.client.get_instruments_info(category=CATEGORY, symbol=s)
                lst = r.get("result", {}).get("list", [])
                if not lst:
                    m[s] = {"qty_step": 0.001, "min_qty": 0.001, "tick": 0.001, "status": ""}
                    continue
                it = lst[0]
                lot = it.get("lotSizeFilter", {}) or {}
                px  = it.get("priceFilter", {}) or {}
                try:
                    qty_step = float(lot.get("qtyStep", "0.001"))
                except Exception:
                    qty_step = 0.001
                try:
                    min_qty  = float(lot.get("minOrderQty", "0.001"))
                except Exception:
                    min_qty = 0.001
                try:
                    tick     = float(px.get("tickSize", "0.001"))
                except Exception:
                    tick = 0.001
                m[s] = {"qty_step": qty_step, "min_qty": min_qty, "tick": tick, "status": it.get("status", "")}
            except Exception as e:
                m[s] = {"qty_step": 0.001, "min_qty": 0.001, "tick": 0.001, "status": ""}
        return m

    

    def _filter_live(self, symbols: List[str]) -> List[str]:
        live: List[str] = []
        for s in symbols:
            try:
                rl_misc.wait()
                r = self.client.get_instruments_info(category=CATEGORY, symbol=s)
                lst = r.get("result", {}).get("list", [])
                if lst:
                    st = str(lst[0].get("status", "")).lower()
                    # Only keep symbols explicitly marked trading/live
                    if st in ("trading", "1", "live"):
                        live.append(s)
                    else:
                        self._skipped.append(s)
                    continue
                # Fallback: if instrument API didn't return info, try kline presence
                try:
                    rl_kline.wait()
                    kr = self.client.get_kline(category=CATEGORY, symbol=s, interval=BYBIT_INTERVAL, limit=2)
                    if kr.get("result", {}).get("list"):
                        live.append(s)
                    else:
                        self._skipped.append(s)
                except Exception:
                    self._skipped.append(s)
            except Exception:
                # If even instruments API errors, treat as skipped to avoid runtime warnings
                self._skipped.append(s)
        return live



    def total_exposure_usdt(self, sym_hint: Optional[str] = None, price_hint: Optional[float] = None) -> float:
        total = 0.0
        for s, t in self.traders.items():
            if getattr(t, "_disabled", False) or t.dir == 0 or t.qty <= 0:
                continue
            if s == sym_hint and price_hint is not None:
                price = price_hint
            else:
                price = t.last_close if t.last_close is not None else (t.avg or 0.0)
            total += abs(t.qty * price)
        return total

    def _safe_get_equity_usdt(self) -> float:
        # Try UNIFIED then CONTRACT without raising
        try:
            rl_misc.wait()
            r = self.client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            lst = (r.get("result", {}) or {}).get("list", []) or []
            if lst:
                coins = (lst[0].get("coin", []) or [])
                if coins:
                    c0 = coins[0]
                    for key in ("equity","walletBalance","availableToWithdraw","availableBalance"):
                        v = c0.get(key)
                        if v is not None:
                            try:
                                return float(v)
                            except Exception:
                                pass
        except Exception:
            pass
        try:
            rl_misc.wait()
            r = self.client.get_wallet_balance(accountType="CONTRACT", coin="USDT")
            lst = (r.get("result", {}) or {}).get("list", []) or []
            if lst:
                coins = (lst[0].get("coin", []) or [])
                if coins:
                    c0 = coins[0]
                    for key in ("equity","walletBalance","availableToWithdraw","availableBalance"):
                        v = c0.get(key)
                        if v is not None:
                            try:
                                return float(v)
                            except Exception:
                                pass
        except Exception:
            pass
        return 0.0

    def _update_hourly_budget_if_due(self, next_close_ts: int):
        # Keep budget refresh hourly
        try:
            now = int(time.time())
            trigger_ts = int(next_close_ts) - 1800  # H:30 relative to the coming bar's close
            hour_marker = int(next_close_ts // 3600)
            if now >= trigger_ts and self._last_budget_hour != hour_marker:
                eq = self._safe_get_equity_usdt()
                if eq > 0.0:
                    self.dynamic_leg_usdt = max(round((eq / 120.0) * LEVERAGE_X, 6), 1e-6)
                    self._last_budget_hour = hour_marker
                    logging.info("[BUDGET] equity=%.6f → dynamic_leg_usdt=%.6f (hour=%d, with leverage %.1fx)", eq, self.dynamic_leg_usdt, hour_marker, LEVERAGE_X)
                else:
                    self._last_budget_hour = hour_marker
                    logging.warning("[BUDGET] equity fetch failed (eq=%.6f); keep dynamic_leg_usdt=%.6f", eq, self.dynamic_leg_usdt)
        except Exception as e:
            logging.warning("[BUDGET] update failed: %s", e)

    def _barclose_pass(self):
        bar_pass_start = time.time()
        # Optional jitter at start
        if START_JITTER_MAX_SEC > 0:
            time.sleep(START_JITTER_MAX_SEC)

        for s, t in list(self.traders.items()):
            if getattr(t, "_disabled", False):
                continue
            try:
                # spread calls across SCAN_SPREAD_SEC after bar close
                if SCAN_SPREAD_SEC > 0:
                    target = bar_pass_start + _spread_offset(s)
                    now = time.time()
                    if now < target:
                        time.sleep(target - now)
                # Startup-guard: only for the first entry on startup
                if self._startup and self.STARTUP_REQUIRE_SIGNAL and t.legs == 0:
                    gsig, _ = _bulls_get_last_closed_sig_for_symbol(s)
                    if gsig == 0:
                        logging.debug("[%s] STARTUP-GUARD(TF=%s): last-closed sig=0 → skip opening new position.", s, TF_LABEL)
                        self._startup_skipped_syms.append(s)
                        continue
                prev_legs = t.legs
                t.step()
                if self._startup and prev_legs == 0 and t.legs > 0:
                    self._startup_opened_syms.append(s)
            except Exception as e:
                now = time.time()
                last = self._err_gate.get(s, 0.0)
                if now - last >= 120.0:
                    self._err_gate[s] = now
                    logging.exception("[%s] step error: %s", s, e)
                else:
                    msg = str(e).splitlines()[0]
                    logging.warning("[%s] step error (muted): %s", s, msg)
        if self._startup:
            skipped = len(self._startup_skipped_syms)
            opened  = len(self._startup_opened_syms)
            total   = len(self.traders)
            self._startup = False
            logging.info("[STARTUP-GUARD] pass summary: skipped=%d (sig=0), opened=%d, total=%d → startup mode off.", skipped, opened, total)

    def loop(self):
        next_close = _next_bar_close()
        if STARTUP_PASS:
            self._barclose_pass()
            for _ in range(int(CLOSE_PASS_RETRIES)):
                time.sleep(CLOSE_PASS_RETRY_GAP)
                self._barclose_pass()

        while True:
            now = time.time()
            self._update_hourly_budget_if_due(next_close)

            if now < next_close - 15.0:
                time.sleep(min(IDLE_POLL_SEC, (next_close - now - 15.0)))
                continue

            # ensure bar finalized
            if now < next_close + CLOSE_GRACE_SEC:
                time.sleep(next_close + CLOSE_GRACE_SEC - now)

            self._barclose_pass()
            for _ in range(int(CLOSE_PASS_RETRIES)):
                time.sleep(CLOSE_PASS_RETRY_GAP)
                self._barclose_pass()

            next_close += int(BARFRAME_SEC)

# ---------- Public API guards (Bybit) ----------
try:
    import requests as _requests
except Exception:
    _requests = None

def __bulls__compute_signal_v5_from_klines(_kl):
    length = 50
    bars = 30
    n = len(_kl)
    if n < max(length, 35):
        return [0]*n
    o = [k[1] for k in _kl]
    h = [k[2] for k in _kl]
    l = [k[3] for k in _kl]
    c = [k[4] for k in _kl]
    highest = [0.0]*n
    lowest  = [0.0]*n
    for i in range(n):
        lo = max(0, i - length + 1)
        highest[i] = max(h[lo:i+1])
        lowest[i]  = min(l[lo:i+1])
    out = [0]*n
    bindex = 0
    sindex = 0
    for i in range(n):
        if i >= 4 and c[i] > c[i-4]:
            bindex += 1
        if i >= 4 and c[i] < c[i-4]:
            sindex += 1
        short_cond = (bindex > bars) and (c[i] < o[i]) and (h[i] >= highest[i])
        long_cond  = (sindex > bars) and (c[i] > o[i]) and (l[i] <= lowest[i])
        if short_cond:
            bindex = 0
            out[i] = -1
        elif long_cond:
            sindex = 0
            out[i] = 1
    return out

def __bulls__last_open_ms(frame_sec: int) -> int:
    now_ms = int(time.time() * 1000)
    fm = int(frame_sec * 1000)
    return (now_ms // fm) * fm - fm  # open time of last CLOSED bar

def _bulls_get_last_closed_sig_for_symbol(sym: str):
    """Return (sig, last_open_ms) for the fixed H1 timeframe; (0, last_open_ms) on failure."""
    try:
        if _requests is None:
            return 0, __bulls__last_open_ms(int(BARFRAME_SEC))
        last_open = __bulls__last_open_ms(int(BARFRAME_SEC))
        _key = (sym, int(last_open))
        _val = _last_closed_sig_cache.get(_key)
        if _val is not None:
            return _val
        start = last_open - 400 * int(BARFRAME_SEC * 1000)
        end   = last_open + int(BARFRAME_SEC * 1000)
        params = {"category": "linear", "symbol": sym, "interval": BYBIT_INTERVAL,
                  "start": str(start), "end": str(end), "limit": "200"}
        url = "https://api.bybit.com/v5/market/kline"
        r = _requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            return 0, last_open
        lst = data.get("result", {}).get("list") or []
        if not lst:
            return 0, last_open
        lst.sort(key=lambda x: int(x[0]))
        kl = [[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in lst]
        idx = { int(row[0]): i for i,row in enumerate(kl) }
        if last_open not in idx:
            return 0, last_open
        sigs = __bulls__compute_signal_v5_from_klines(kl)
        _last_closed_sig_cache[_key] = (int(sigs[idx[last_open]]), last_open)
        return int(sigs[idx[last_open]]), last_open
    except Exception:
        return 0, __bulls__last_open_ms(int(BARFRAME_SEC))

def _bulls_get_sig_seq_for_last_n(sym: str, n: int):
    """Return list of last n CLOSED H1 bar signals (ascending)."""
    try:
        if _requests is None or n <= 0:
            return []
        last_open = __bulls__last_open_ms(int(BARFRAME_SEC))
        start   = last_open - max(400, n+5) * int(BARFRAME_SEC * 1000)
        end     = last_open + int(BARFRAME_SEC * 1000)
        url     = "https://api.bybit.com/v5/market/kline"
        params  = {"category":"linear","symbol":sym,"interval":BYBIT_INTERVAL,
                   "start":str(start),"end":str(end),"limit":"200"}
        r = _requests.get(url, params=params, timeout=10); r.raise_for_status()
        data = r.json()
        lst  = (data.get("result",{}) or {}).get("list") or []
        if not lst: return []
        lst.sort(key=lambda x: int(x[0]))
        kl = [[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in lst]
        sigs = __bulls__compute_signal_v5_from_klines(kl)
        idx = { int(row[0]): i for i,row in enumerate(kl) }
        if int(last_open) not in idx: return []
        i_last = idx[int(last_open)]
        i_from = max(0, i_last - (n - 1))
        return [int(sigs[i]) for i in range(i_from, i_last + 1)]
    except Exception:
        return []

# ---------- Main ----------
if __name__ == "__main__":
    syms = read_pairs(PAIRS_FILE)
    if not syms:
        raise SystemExit(f"No symbols loaded from {PAIRS_FILE}")
    MultiBot(syms).loop()
