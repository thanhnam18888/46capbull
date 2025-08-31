# bot_multi.py — Multi-symbol DCA bot for Bybit linear perp (Render)
# Features:
# - Signal matches 18.py (bar-close Lelec): length=50, bars=30, with O/C checks & indices
# - Uses only CLOSED H1 candles (drop the latest in-progress bar)
# - TP maker-first (PostOnly) with TTL and market fallback; Emergency SL (market)
# - MAX_DCA=-1 default (unlimited); Flip-on-profit; Geo-scaling per leg
# - Cross-budget control via total exposure
# - Per-symbol lot filters (qtyStep/minQty/tick); Soft live filter (status + kline fallback)
# - Anti-spam logging (LOG_LEVEL, LOG_VERBOSE_INIT, LOG_THROTTLE_SEC)
# - Throttled API calls; Fail-fast if BYBIT_API_KEY/SECRET missing
# - Bar-close scheduler to fire orders a few seconds after H1 close (no ~:05 delays)
# - STARTUP_PASS: run an immediate bar-close pass at startup so you can test at 04:20
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
LEG_USDT         = env_float("LEG_USDT", 15.0)
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

# Anti-spam logging controls
LOG_LEVEL        = env_str("LOG_LEVEL", "INFO").upper()
LOG_VERBOSE_INIT = env_int("LOG_VERBOSE_INIT", 0) == 1      # per-symbol init chatter
LOG_THROTTLE_SEC = env_float("LOG_THROTTLE_SEC", 30.0)      # per-symbol error throttle

# Bar-close scheduler
BARFRAME_SEC              = env_float("BARFRAME_SEC", 3600.0)
CLOSE_GRACE_SEC           = env_float("CLOSE_GRACE_SEC", 2.0)
CLOSE_PASS_RETRIES        = env_int("CLOSE_PASS_RETRIES", 2)
CLOSE_PASS_RETRY_GAP      = env_float("CLOSE_PASS_RETRY_GAP", 1.0)
IDLE_POLL_SEC             = env_float("IDLE_POLL_SEC", 10.0)
STARTUP_PASS              = env_int("STARTUP_PASS", 1)

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

rl_switch = RateLimiter(0.18)
rl_lev    = RateLimiter(0.18)
rl_kline  = RateLimiter(0.08)
rl_misc   = RateLimiter(0.06)

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

        # TP maker-first state
        self.tp_pending = False
        self.tp_px: Optional[float] = None
        self.tp_deadline = 0.0

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

    def ticker(self) -> Tuple[float, float, float]:
        rl_misc.wait()
        r = self.client.get_tickers(category=CATEGORY, symbol=self.symbol)
        px = float(r["result"]["list"][0]["lastPrice"])
        return (px, px, px)

    def klines_1h(self, limit: int = 200) -> List[List[float]]:
        rl_kline.wait()
        r = self.client.get_kline(category=CATEGORY, symbol=self.symbol, interval="60", limit=limit)
        out: List[List[float]] = []
        for it in r["result"]["list"]:
            ts = int(it[0]); o = float(it[1]); h = float(it[2]); l = float(it[3]); c = float(it[4])
            out.append([ts, o, h, l, c])
        out.sort(key=lambda x: x[0])
        return out

    def _calc_leg_qty(self, price: float, dir_sign: int) -> float:
        idx = self.legs
        scale = VOL_SCALE_LONG if dir_sign > 0 else VOL_SCALE_SHORT
        leg_usdt = LEG_USDT * (scale ** idx)

        if CROSS_BUDGET_USDT > 0.0:
            used = self.ctx.total_exposure_usdt(self.symbol, price_hint=price)
            left = CROSS_BUDGET_USDT * (1.0 - RESERVE_PCT) - used
            if left <= 0:
                return 0.0
            leg_usdt = min(leg_usdt, left)

        qty = leg_usdt / price
        qty = round_step_floor(qty, self.qty_step)
        return max(qty, self.min_qty)

    def place_market(self, side: str, qty: float, reduce_only: bool = False):
        if qty <= 0:
            return
        rl_misc.wait()
        self.client.place_order(category=CATEGORY, symbol=self.symbol, side=side,
                                orderType="Market", qty=str(qty),
                                reduceOnly=reduce_only, timeInForce="IOC")

    def cancel_all_orders(self):
        try:
            rl_misc.wait()
            self.client.cancel_all_orders(category=CATEGORY, symbol=self.symbol)
        except Exception as e:
            logging.debug("[%s] cancel_all_orders failed: %s", self.symbol, e)

    def place_limit(self, side: str, qty: float, price: float, reduce_only: bool = False, post_only: bool = False):
        if qty <= 0:
            return None
        rl_misc.wait()
        tif = "PostOnly" if post_only else "GoodTillCancel"
        try:
            r = self.client.place_order(category=CATEGORY, symbol=self.symbol, side=side,
                                        orderType="Limit", qty=str(qty), price=str(price),
                                        reduceOnly=reduce_only, timeInForce=tif)
            return r.get("result", {}).get("orderId")
        except Exception as e:
            logging.warning("[%s] place_limit failed: %s", self.symbol, e)
            return None

    def open_leg(self, dir_sign: int, price: float):
        qty = self._calc_leg_qty(price, dir_sign)
        if qty <= 0:
            logging.info("[%s] skip open leg (qty=0 due to budget/min_qty)", self.symbol)
            return
        side = "Buy" if dir_sign > 0 else "Sell"
        self.place_market(side, qty, reduce_only=False)

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

        # add only the new leg's taker fee (keep 0 by default)
        self.entry_fees += qty * price * TAKER_FEE
        if self.funding_anchor is None:
            self.funding_anchor = time.time()

        # reset TP pending after position change
        self.tp_pending = False
        self.tp_px = None
        self.tp_deadline = 0.0

        logging.info("[%s] OPEN leg #%d dir=%+d qty=%.6f @ %.6f (avg=%.6f)",
                     self.symbol, self.legs, self.dir, qty, price, self.avg)

    def close_all(self, price: float):
        if self.dir == 0 or self.qty <= 0:
            return
        side = "Sell" if self.dir > 0 else "Buy"
        self.place_market(side, self.qty, reduce_only=True)
        logging.info("[%s] CLOSE ALL qty=%.6f @ %.6f (avg=%.6f)", self.symbol, self.qty, price, self.avg)
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

    def unreal_pnl(self, price: float) -> float:
        if self.dir == 0 or self.qty <= 0 or self.avg is None:
            return 0.0
        gross = (price / self.avg - 1.0) if self.dir > 0 else (1.0 - price / self.avg)
        px_pnl = self.qty * self.avg * gross
        exit_fee = self.qty * price * TAKER_FEE
        return px_pnl - (self.entry_fees + exit_fee + self.funding_paid)

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
        kl = self.klines_1h(limit=200)
        # Use only fully CLOSED bars: drop the newest bar which may be in-progress
        if not kl or len(kl) < 61:
            logging.debug("[%s] insufficient klines; skip", self.symbol)
            return
        kl_closed = kl[:-1]
        last_ts = int(kl_closed[-1][0])
        if self.last_bar_ts == last_ts:
            return
        self.last_bar_ts = last_ts

        # 18.py-matching signal
        sig = bulls_signal_from_klines_barclose(kl_closed)[-1]

        _, _, price = self.ticker()

        self._funding_tick(price)

        # TP maker-first with TTL; SL market
        if self.dir != 0 and self.qty > 0 and self.avg is not None:
            if TP_PCT > 0.0:
                want_tp = self.avg * (1.0 + self.dir * TP_PCT)
                if not self.tp_pending:
                    reached = (self.dir > 0 and price >= want_tp) or (self.dir < 0 and price <= want_tp)
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
                            self.close_all(price)
                            return
                else:
                    if time.time() >= self.tp_deadline:
                        self.cancel_all_orders()
                        self.close_all(price)
                        self.tp_pending = False
                        self.tp_px = None
                        self.tp_deadline = 0.0
                        return

            if EMERGENCY_SL_PCT > 0.0:
                sl_px = self.avg * (1.0 - self.dir * EMERGENCY_SL_PCT)
                hit = (self.dir > 0 and price <= sl_px) or (self.dir < 0 and price >= sl_px)
                if hit:
                    if self.tp_pending:
                        self.cancel_all_orders()
                        self.tp_pending = False
                        self.tp_px = None
                        self.tp_deadline = 0.0
                    self.close_all(price)
                    return

        if self.dir == 0:
            if sig != 0:
                self.open_leg(sig, price)
            return

        upnl = self.unreal_pnl(price)

        if sig == self.dir:
            if upnl < 0.0:
                if MAX_DCA < 0 or self.legs < 1 + MAX_DCA:
                    self.open_leg(self.dir, price)
                else:
                    logging.info("[%s] MAX_DCA reached; skip DCA", self.symbol)
        elif sig == -self.dir:
            if self.legs == 1:
                self.close_all(price)
                self.open_leg(-self.dir, price)
                return
            else:
                if upnl >= 0.0 or (FLIP_ON_PROFIT and upnl > 0.0):
                    self.close_all(price)
                    self.open_leg(-self.dir, price)
                    return

        if self.legs > 1 and upnl >= 0.0 and sig == 0:
            self.close_all(price)

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

        if self._skipped:
            preview = ", ".join(self._skipped[:20])
            more = "" if len(self._skipped) <= 20 else f" (+{len(self._skipped)-20} more)"
            logging.info("Skipped %d non-live symbols: %s%s", len(self._skipped), preview, more)

        logging.info(
            "MultiBot started for %d symbols; LEG_USDT=%.4f, lev=%.1fx, fee=%.4f, funding_8h=%.6f",
            len(self.traders), LEG_USDT, LEVERAGE_X, TAKER_FEE, FUNDING_8H
        )

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
                # 1) Try instrument status on linear
                rl_misc.wait()
                r = self.client.get_instruments_info(category=CATEGORY, symbol=s)
                lst = r.get("result", {}).get("list", [])
                st  = str(lst[0].get("status", "")).lower() if lst else ""

                if st in ("trading", "1", "live"):
                    live.append(s)
                    continue

                # 2) Fallback: if kline returns data for linear, treat as live
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
                # API hiccup -> allow symbol; runtime will handle errors later
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
                try:
                    _, _, price = t.ticker()
                except Exception:
                    price = t.avg or 0.0
            total += abs(t.qty * price)
        return total

    def _barclose_pass(self):
        # One pass across all symbols to evaluate bar-close signals.
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
        # Align to next H1 close
        next_close = _next_bar_close()
        # Immediate bar-close pass on startup for quick testing (e.g., deploy at 04:20)
        if STARTUP_PASS:
            self._barclose_pass()
            for _ in range(int(CLOSE_PASS_RETRIES)):
                time.sleep(CLOSE_PASS_RETRY_GAP)
                self._barclose_pass()

        while True:
            now = time.time()
            # Far from close -> idle sleep
            if now < next_close - 15.0:
                time.sleep(min(IDLE_POLL_SEC, (next_close - now - 15.0)))
                continue

            # Wait until just after close to ensure the last bar is finalized on the API
            if now < next_close + CLOSE_GRACE_SEC:
                time.sleep(next_close + CLOSE_GRACE_SEC - now)
                now = time.time()

            # Run the pass at bar close (+grace). Do quick retries if some symbols' klines are late.
            self._barclose_pass()
            for _ in range(int(CLOSE_PASS_RETRIES)):
                time.sleep(CLOSE_PASS_RETRY_GAP)
                self._barclose_pass()

            # Move to next bar close
            next_close += int(BARFRAME_SEC)

# ---------- Main ----------
if __name__ == "__main__":
    syms = read_pairs(PAIRS_FILE)
    if not syms:
        raise SystemExit(f"No symbols loaded from {PAIRS_FILE}")
    MultiBot(syms).loop()
