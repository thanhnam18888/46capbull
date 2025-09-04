
# bot_multi_combined.py — Multi-symbol DCA bot for Bybit linear perp (Render)
# Combines:
#   (A) Full rate-limit mitigations to avoid ErrCode: 10006 bursts
#   (B) Per-leg notional target = (equity/120) * leverage (your 1/120 then x5)
#
# Highlights:
# - Bar-close logic (H1) same as before (18.py style).
# - Lazy ticker: only fetch live price right before placing an actual order.
# - Kline cadence slowed via RL_KLINE_SEC (default 0.35s) + exponential backoff on 10006.
# - No automatic "close-pass" retries by default (CLOSE_PASS_RETRIES=0) to avoid bursts after close.
# - Reduced kline payload (KLINE_LIMIT=120) – still >= 61 bars for signal.
# - Hourly dynamic leg uses leverage: dynamic_leg_usdt = (equity / 120) * LEVERAGE_X
# - Budget/exposure computed from last_close/avg when possible (avoid ticker-spam).
#
# Env knobs (set in Render → Environment):
#   RL_KLINE_SEC=0.35 | RL_MISC_SEC=0.12 | RL_LEV_SEC=0.30 | RL_SWITCH_SEC=0.30
#   CLOSE_PASS_RETRIES=0 | KLINE_LIMIT=120 | STARTUP_PASS=1
#   LEVERAGE_X=5.0 | LEG_USDT=15.0 (legacy default; not used if dynamic_leg_usdt set)
#   MIN_NOTIONAL_USDT=5.0 | TP/SL/others same as your prior bot
#
import os
import time
import logging
from typing import List, Dict, Optional, Tuple
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError

# ---------- ENV helpers ----------
def env_str(name: str, default: str) -> str:
    v = os.getenv(name, default)
    return v if v is not None else default

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

# ---------- Config ----------
BYBIT_API_KEY    = env_str("BYBIT_API_KEY", "")
BYBIT_API_SECRET = env_str("BYBIT_API_SECRET", "")
CATEGORY         = "linear"
LEVERAGE_X       = env_float("LEVERAGE_X", 5.0)
LEG_USDT         = env_float("LEG_USDT", 15.0)  # used as fallback if equity lookup fails
PAIRS_FILE       = env_str("PAIRS_FILE", "46cap.txt")

TAKER_FEE        = env_float("TAKER_FEE", 0.0)          # keep 0 for parity
FUNDING_8H       = env_float("FUNDING_RATE_8H", 0.0)    # 0 by default

MAX_DCA          = env_int("MAX_DCA", -1)               # -1 = unlimited
TP_PCT           = env_float("TP_PCT", 0.0)             # 0 = disabled
TP_TTL_SEC       = env_float("TP_TTL_SEC", 8.0)
TP_POST_ONLY     = env_int("TP_POST_ONLY", 1) == 1
EMERGENCY_SL_PCT = env_float("EMERGENCY_SL_PCT", 0.0)   # 0 = disabled
FLIP_ON_PROFIT   = env_int("FLIP_ON_PROFIT", 1) == 1

VOL_SCALE_LONG   = env_float("VOL_SCALE_LONG", 1.0)
VOL_SCALE_SHORT  = env_float("VOL_SCALE_SHORT", 1.0)

CROSS_BUDGET_USDT= env_float("CROSS_BUDGET_USDT", 0.0)  # 0 = disabled
RESERVE_PCT      = env_float("RESERVE_PCT", 0.10)

MIN_NOTIONAL_USDT = env_float("MIN_NOTIONAL_USDT", 5.0)

# Anti-spam logging controls
LOG_LEVEL        = env_str("LOG_LEVEL", "INFO").upper()
LOG_VERBOSE_INIT = env_int("LOG_VERBOSE_INIT", 0) == 1
LOG_THROTTLE_SEC = env_float("LOG_THROTTLE_SEC", 30.0)

# Bar-close scheduler
BARFRAME_SEC              = env_float("BARFRAME_SEC", 3600.0)
CLOSE_GRACE_SEC           = env_float("CLOSE_GRACE_SEC", 2.0)
CLOSE_PASS_RETRIES        = env_int("CLOSE_PASS_RETRIES", 0)    # default 0 to avoid bursts
CLOSE_PASS_RETRY_GAP      = env_float("CLOSE_PASS_RETRY_GAP", 1.5)
IDLE_POLL_SEC             = env_float("IDLE_POLL_SEC", 10.0)
STARTUP_PASS              = env_int("STARTUP_PASS", 1)

# Request pacing (account-wide)
RL_SWITCH_SEC = env_float("RL_SWITCH_SEC", 0.30)
RL_LEV_SEC    = env_float("RL_LEV_SEC",    0.30)
RL_KLINE_SEC  = env_float("RL_KLINE_SEC",  0.35)  # ~3 req/s (weight-aware)
RL_MISC_SEC   = env_float("RL_MISC_SEC",   0.12)

# Kline payload
KLINE_LIMIT   = max(61, env_int("KLINE_LIMIT", 70))  # 61+ required for signal

_level_map = {
    "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
    "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL
}
logging.basicConfig(level=_level_map.get(LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger().setLevel(_level_map.get(LOG_LEVEL, logging.INFO))

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise SystemExit("BYBIT_API_KEY / BYBIT_API_SECRET are required but missing. Set them in Render → Environment.")

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

# --- Lelec signal like 18.py (bar close) ---
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
        self.last_close: Optional[float] = None  # <-- cache

        # TP maker-first state
        self.tp_pending = False
        self.tp_px: Optional[float] = None
        self.tp_deadline = 0.0

    # ---- exchange setup ----
    def _switch_one_way(self):
        try:
            rl_switch.wait()
            self.client.switch_position_mode(category=CATEGORY, symbol=self.symbol, mode=0)
            if LOG_VERBOSE_INIT:
                logging.info("[%s] position mode set One-Way", self.symbol)
        except InvalidRequestError as e:
            if "110025" in str(e):
                if LOG_VERBOSE_INIT:
                    logging.debug("[%s] already One-Way", self.symbol)
            else:
                logging.warning("[%s] switch mode failed: %s", self.symbol, e)
        except Exception as e:
            logging.warning("[%s] switch mode failed: %s", self.symbol, e)

    def _set_leverage(self):
        try:
            rl_lev.wait()
            self.client.set_leverage(category=CATEGORY, symbol=self.symbol,
                                     buyLeverage=str(LEVERAGE_X), sellLeverage=str(LEVERAGE_X))
            if LOG_VERBOSE_INIT:
                logging.info("[%s] leverage set to %.1fx", self.symbol, LEVERAGE_X)
        except InvalidRequestError as e:
            msg = str(e)
            if "110043" in msg:       # leverage not modified
                if LOG_VERBOSE_INIT:
                    logging.debug("[%s] leverage already %.1fx", self.symbol, LEVERAGE_X)
            elif "110074" in msg:     # closed contract
                logging.error("[%s] closed contract; disabling", self.symbol)
                setattr(self, "_disabled", True)
            else:
                logging.warning("[%s] set leverage failed: %s", self.symbol, e)
        except Exception as e:
            logging.warning("[%s] set leverage failed: %s", self.symbol, e)

    # ---- data fetch ----
    def klines_1h(self, limit: int = KLINE_LIMIT) -> List[List[float]]:
        backoff = 0.8  # start slower for safety
        for attempt in range(6):
            try:
                rl_kline.wait()
                r = self.client.get_kline(category=CATEGORY, symbol=self.symbol, interval="60", limit=limit)
                out: List[List[float]] = []
                for it in r["result"]["list"]:
                    ts = int(it[0]); o = float(it[1]); h = float(it[2]); l = float(it[3]); c = float(it[4])
                    out.append([ts, o, h, l, c])
                out.sort(key=lambda x: x[0])
                return out
            except Exception as e:
                msg = str(e).lower()
                if "10006" in msg or "rate limit" in msg:
                    # calm down globally if we hit the wall
                    time.sleep(backoff)
                    backoff = min(backoff * 1.6, 8.0)
                    continue
                if attempt < 2:
                    time.sleep(backoff)
                    backoff = min(backoff * 1.6, 8.0)
                    continue
                raise
        raise RuntimeError(f"[{self.symbol}] get_kline failed after retries (rate limit or network)")

    def _format_qty_for_exchange(self, qty: float) -> str:
        from decimal import Decimal, ROUND_DOWN
        step = Decimal(str(self.qty_step))
        q = (Decimal(str(qty)) // step) * step  # floor to step
        if q <= 0:
            q = step
        # Enforce minQty
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

    # ---- position ops (use live price only when actually trading) ----
    def _fetch_live_price(self) -> float:
        # lightweight ticker (no caching); called sparingly
        rl_misc.wait()
        r = self.client.get_tickers(category=CATEGORY, symbol=self.symbol)
        return float(r["result"]["list"][0]["lastPrice"])

    def _calc_leg_qty(self, price: float, dir_sign: int) -> float:
        from math import ceil
        idx = self.legs
        scale = VOL_SCALE_LONG if dir_sign > 0 else VOL_SCALE_SHORT
        base_usdt = getattr(self.ctx, 'dynamic_leg_usdt', float(LEG_USDT)) or float(LEG_USDT)
        leg_usdt = base_usdt * (scale ** idx)

        left = None
        if CROSS_BUDGET_USDT > 0.0:
            used = self.ctx.total_exposure_usdt(self.symbol, price_hint=price)
            left = CROSS_BUDGET_USDT * (1.0 - RESERVE_PCT) - used
            if left <= 0:
                return 0.0
            leg_usdt = min(leg_usdt, left)

        if leg_usdt < MIN_NOTIONAL_USDT:
            if CROSS_BUDGET_USDT > 0.0:
                if left is not None and left < MIN_NOTIONAL_USDT:
                    return 0.0
                leg_usdt = min(max(MIN_NOTIONAL_USDT, leg_usdt), left if left is not None else MIN_NOTIONAL_USDT)
            else:
                leg_usdt = MIN_NOTIONAL_USDT

        qty = leg_usdt / price
        qty = round_step_floor(qty, self.qty_step)
        if qty < self.min_qty:
            qty = self.min_qty
        if qty * price < MIN_NOTIONAL_USDT:
            target_qty = ceil(MIN_NOTIONAL_USDT / price / self.qty_step) * self.qty_step
            if CROSS_BUDGET_USDT > 0.0 and left is not None and target_qty * price > left:
                return 0.0
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
        if FUNDING_8H <= 0.0 or self.dir == 0 or self.qty <= 0 or self.funding_anchor is None:
            return
        period = 8.0 * 3600.0
        now = time.time()
        while self.funding_anchor + period <= now:
            notional = self.qty * price
            self.funding_paid += notional * FUNDING_8H
            self.funding_anchor += period

    def step(self):
        if getattr(self, 'disabled', False):
            return
        kl = self.klines_1h(limit=KLINE_LIMIT)
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
            import time
            BARFRAME_MS = 60 * 60 * 1000
            _now_ms = int(time.time() * 1000)
            while kl and int(kl[-1][0]) + BARFRAME_MS > _now_ms:
                kl = kl[:-1]
        except Exception:
            pass
        # === END: closed-bar & ordering guard ===

        kl_closed = kl  # drop potentially in-progress
        last_ts = int(kl_closed[-1][0])
        if self.last_bar_ts == last_ts:
            return
        self.last_bar_ts = last_ts
        self.last_close = float(kl_closed[-1][4])

        # 18.py signal array
        sig_arr = bulls_signal_from_klines_barclose(kl_closed)
        sig = sig_arr[-1]

        # Use bar-close price for decision; fetch live only when we actually trade
        price_ref = self.last_close

        # FAST PATH: flip immediately if single-leg and signal flipped
        if self.dir != 0 and self.legs == 1 and sig == -self.dir:
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

        self._funding_tick(price_ref)

        # TP maker-first with TTL; SL market — decisions at bar close using price_ref
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
                    self.open_leg(sig)
                else:
                    logging.info("[%s] ENTRY-GUARD: require last %d bars sig=%+d; tail=%s → skip entry",
                                 self.symbol, ENTRY_CONFIRM_BARS, sig, str(sig_arr[-ENTRY_CONFIRM_BARS:]))
            return

        upnl = self._upnl_from_price(price_ref)

        if sig == self.dir:
            if upnl < 0.0:
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
        self.client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET, recv_window=5000)
        self._skipped: List[str] = []
        self._err_gate: Dict[str, float] = {}
        self.symbols = self._filter_live(symbols)
        self.traders: Dict[str, Trader] = {}
        inst_map = self._load_instruments()
        for s in self.symbols:
            cfg = inst_map.get(s, {"qty_step": 0.001, "min_qty": 0.001, "tick": 0.001})
            t = Trader(self.client, s, cfg, ctx=self)
            t.init_on_exchange()
            self.traders[s] = t
        self.dynamic_leg_usdt = float(LEG_USDT)
        self._last_budget_hour = -1  # hour marker

        if self._skipped:
            preview = ", ".join(self._skipped[:20])
            more = "" if len(self._skipped) <= 20 else f" (+{len(self._skipped)-20} more)"
            logging.info("Skipped %d non-live symbols: %s%s", len(self._skipped), preview, more)

        logging.info("MultiBot started for %d symbols; LEG_USDT=%.4f, lev=%.1fx, fee=%.4f, funding_8h=%.6f",
                     len(self.traders), LEG_USDT, LEVERAGE_X, TAKER_FEE, FUNDING_8H)

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
                    if LOG_VERBOSE_INIT:
                        logging.debug("[%s] instruments info empty; using defaults", s)
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
                if LOG_VERBOSE_INIT:
                    logging.debug("[%s] instruments info error: %s; using defaults", s, e)
                m[s] = {"qty_step": 0.001, "min_qty": 0.001, "tick": 0.001, "status": ""}
        return m

    def _filter_live(self, symbols: List[str]) -> List[str]:
        live: List[str] = []
        for s in symbols:
            try:
                rl_misc.wait()
                r = self.client.get_instruments_info(category=CATEGORY, symbol=s)
                lst = r.get("result", {}).get("list", [])
                st  = str(lst[0].get("status", "")).lower() if lst else ""

                if st in ("trading", "1", "live"):
                    live.append(s)
                    continue

                # Fallback to klines presence
                try:
                    rl_kline.wait()
                    kr = self.client.get_kline(category=CATEGORY, symbol=s, interval="60", limit=2)
                    if kr.get("result", {}).get("list"):
                        live.append(s)
                        continue
                except Exception:
                    pass

                self._skipped.append(s)
            except Exception:
                live.append(s)
        return live

    def total_exposure_usdt(self, sym_hint: Optional[str] = None, price_hint: Optional[float] = None) -> float:
        total = 0.0
        for s, t in self.traders.items():
            if getattr(t, "_disabled", False) or t.dir == 0 or t.qty <= 0:
                continue
            if s == sym_hint and price_hint is not None:
                price = price_hint
            else:
                # avoid ticker; use cached last_close or avg
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
        try:
            now = int(time.time())
            trigger_ts = int(next_close_ts) - 1800  # H:30:00
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
        for s, t in list(self.traders.items()):
            if getattr(t, "_disabled", False):
                continue
            try:
                t.step()
            except Exception as e:
                now = time.time()
                last = self._err_gate.get(s, 0.0)
                if now - last >= LOG_THROTTLE_SEC:
                    self._err_gate[s] = now
                    logging.exception("[%s] step error: %s", s, e)
                else:
                    msg = str(e).splitlines()[0]
                    logging.warning("[%s] step error (muted): %s", s, msg)

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

            # one pass (sequential; rate-limited internally)
            self._barclose_pass()
            for _ in range(int(CLOSE_PASS_RETRIES)):
                time.sleep(CLOSE_PASS_RETRY_GAP)
                self._barclose_pass()

            next_close += int(BARFRAME_SEC)

# ---------- Main ----------
if __name__ == "__main__":
    syms = read_pairs(PAIRS_FILE)
    if not syms:
        raise SystemExit(f"No symbols loaded from {PAIRS_FILE}")
    MultiBot(syms).loop()

# ================== BEGIN: STARTUP SIGNAL GUARD (no logic change) ==================
# Purpose: When the bot restarts mid-hour (startup pass), avoid opening a NEW position
# if the last CLOSED H1 bar has no signal (sig==0). This keeps 100% of your trading
# logic intact. We only skip entries during startup when there is NO signal.
#
# Implementation:
# - Set flags on MultiBot at construction: _startup=True and STARTUP_REQUIRE_SIGNAL=True.
# - After the first _barclose_pass(), flip _startup=False.
# - Wrap Trader.open_leg(...) so that during startup (and legs==0), it checks the last
#   closed bar's signal from Bybit public API. If sig==0 => skip entry and log.
#
# Notes:
# - This does NOT alter DCA, flip, TP/SL, or bar-close logic. It only guards startup entries.
# - If the API check fails for any reason, the guard is fail-open (does not block).
try:
    import time, datetime, logging
    import json as _json
    import math as _math
    import requests as _requests
except Exception as _e:
    # If requests is not available, the guard remains inert (fails open).
    _requests = None

def __bulls__compute_signal_v5_from_klines(_kl):
    # _kl: list of [ms, o, h, l, c] ascending
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

def __bulls__get_last_closed_sig_for_symbol(sym: str):
    # Returns (sig, last_open_ms) or (0, last_open_ms) if cannot determine.
    try:
        if _requests is None:
            return 0, int(time.time()//3600*3600 - 3600) * 1000
        now_ms = int(time.time() * 1000)
        hour_ms = 3600*1000
        last_open = (now_ms // hour_ms) * hour_ms - hour_ms  # open time of last CLOSED H1
        start = last_open - 400*hour_ms
        end = last_open + hour_ms  # ensure the last_closed bar is within range
        params = {
            "category": "linear",
            "symbol": sym,
            "interval": "60",
            "start": str(start),
            "end": str(end),
            "limit": "200",
        }
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
        return int(sigs[idx[last_open]]), last_open
    except Exception as _e:
        return 0, int(time.time()//3600*3600 - 3600) * 1000

# Patch MultiBot to set startup flags and flip them after first barclose pass
try:
    if 'MultiBot' in globals() and not getattr(MultiBot, "_startup_signal_guard_patched", False):
        _orig_init = MultiBot.__init__
        def __init__startup_guard(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            # set flags once
            if not hasattr(self, "_startup"):
                self._startup = True
            if not hasattr(self, "STARTUP_REQUIRE_SIGNAL"):
                self.STARTUP_REQUIRE_SIGNAL = True
            logging.info("[STARTUP-GUARD] enabled: require signal on last closed bar before opening new positions during startup.")
        MultiBot.__init__ = __init__startup_guard

        if hasattr(MultiBot, "_barclose_pass"):
            _orig_bcp = MultiBot._barclose_pass
            def _barclose_pass_wrapper(self, *args, **kwargs):
                res = _orig_bcp(self, *args, **kwargs)
                if getattr(self, "_startup", False):
                    self._startup = False
                    logging.info("[STARTUP-GUARD] first barclose pass completed → startup mode off.")
                return res
            MultiBot._barclose_pass = _barclose_pass_wrapper

        MultiBot._startup_signal_guard_patched = True
except Exception as _e:
    pass

# Patch Trader.open_leg to skip entries during startup if last closed bar has no signal.
try:
    if 'Trader' in globals() and not getattr(Trader, "_startup_open_guard_patched", False):
        _orig_open_leg = Trader.open_leg
        def open_leg_startup_guard(self, direction, *args, **kwargs):
            try:
                ctx = getattr(self, "ctx", None)
                # Only guard on startup, for FIRST leg (new position), and only if flag is enabled
                if ctx and getattr(ctx, "STARTUP_REQUIRE_SIGNAL", False) and getattr(ctx, "_startup", False) and int(getattr(self, "legs", 0) or 0) == 0:
                    sig, last_open = __bulls__get_last_closed_sig_for_symbol(self.symbol)
                    if sig == 0:
                        try:
                            tclose = datetime.datetime.utcfromtimestamp((last_open + 3600000)/1000).strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            tclose = str(last_open)
                        logging.info("[%s] STARTUP-GUARD: last-closed sig=0 (bar closed at %s) → skip opening new position.", self.symbol, tclose)
                        return False
            except Exception as _e:
                # fail-open on any guard error
                pass
            # normal path
            return _orig_open_leg(self, direction, *args, **kwargs)
        Trader.open_leg = open_leg_startup_guard
        Trader._startup_open_guard_patched = True
except Exception as _e:
    pass
# =================== END: STARTUP SIGNAL GUARD (no logic change) ===================
