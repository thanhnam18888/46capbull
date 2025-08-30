
# bot_multi.py — Multi-symbol Bulls DCA bot (Render) — auto-reverse like 18.py
# Core rules (ignoring fees for breakeven logic, per user request):
# - At each closed 1H bar, read signal s ∈ {-1,0,+1} from Lelec(50/30).
# - If Flat and s!=0: OPEN 1 leg (LEG_USDT) in direction s.
# - If In Position and s==dir and unrealized PnL<0: DCA add 1 leg.
# - If Opposite signal:
#     * if legs==1: CLOSE then immediately OPEN opposite (auto-reverse same bar).
#     * if legs>1: CLOSE & reverse only if breakeven (PnL>=0).
# - If No signal and legs>1 and breakeven: CLOSE now (breakeven exit).
#
# Extras included:
# - Unlimited DCA by default (MAX_DCA<0). ENV still supported but default is -1 (=unlimited).
# - Flip-on-profit (ENV FLIP_ON_PROFIT=1).
# - Geo-scaling per leg (VOL_SCALE_LONG/SHORT, default 1.0).
# - Cross-budget cap via CROSS_BUDGET_USDT & RESERVE_PCT (optional; set to 0 to disable).
# - Throttled API calls to avoid 10006 rate limit.
# - Filter non-live instruments (e.g., closed contracts) to avoid empty klines.
#
# Note: This file only uses TP/SL "soft" (logic-level) closes. No reduce-only/stop trigger orders are placed.
# If you want "hard" TP/SL orders on-exchange, we can extend to create/cancel conditional orders accordingly.

import os, time, math, hmac, hashlib, logging
from typing import List, Dict, Optional, Tuple

from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError

# ---------------------- ENV ----------------------
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

BYBIT_API_KEY    = env_str("BYBIT_API_KEY", "")
BYBIT_API_SECRET = env_str("BYBIT_API_SECRET", "")
CATEGORY         = "linear"
LEVERAGE_X       = env_float("LEVERAGE_X", 5.0)
LEG_USDT         = env_float("LEG_USDT", 15.0)
ORDER_MODE       = env_str("ORDER_MODE", "MARKET").upper()  # MARKET only in this clean build
PAIRS_FILE       = env_str("PAIRS_FILE", "46cap.txt")
POLL_SEC         = env_float("POLL_SEC", 2.5)

# PnL/fees/funding controls
TAKER_FEE        = env_float("TAKER_FEE", 0.0)          # set 0 for parity with 18.py
FUNDING_8H       = env_float("FUNDING_RATE_8H", 0.0)    # 0 by default to ignore funding

# Enhancements
MAX_DCA          = env_int("MAX_DCA", -1)     # -1 = unlimited
TP_PCT           = env_float("TP_PCT", 0.0)   # TP percent (e.g., 0.002 = 0.2%)
TP_TTL_SEC       = env_float("TP_TTL_SEC", 8.0)  # wait seconds for maker fill before fallback
TP_POST_ONLY     = env_int("TP_POST_ONLY", 1) == 1
EMERGENCY_SL_PCT = env_float("EMERGENCY_SL_PCT", 0.0)  # soft SL; 0 = disabled
FLIP_ON_PROFIT   = env_int("FLIP_ON_PROFIT", 1) == 1
VOL_SCALE_LONG   = env_float("VOL_SCALE_LONG", 1.0)
VOL_SCALE_SHORT  = env_float("VOL_SCALE_SHORT", 1.0)
CROSS_BUDGET_USDT= env_float("CROSS_BUDGET_USDT", 0.0)  # 0 = disabled
RESERVE_PCT      = env_float("RESERVE_PCT", 0.10)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------------- Small Throttler ----------------------
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

rl_switch   = RateLimiter(0.18)
rl_lev      = RateLimiter(0.18)
rl_kline    = RateLimiter(0.08)
rl_misc     = RateLimiter(0.06)

# ---------------------- Utilities ----------------------
def read_pairs(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip().upper()
            if not s:
                continue
            if not s.endswith("USDT"):
                s = s + "USDT"
            out.append(s)
    return out

def round_step(x: float, step: float) -> float:
    if step <= 0: return x
    k = round(x / step)
    return max(step, k * step)

def lelec_signal_from_klines(kl: List[List[float]], n: int = 50, minbars: int = 30) -> int:
    """
    Compute Lelec-like extremum signal using last n highs/lows with inner window minbars.
    Returns +1 (bull), -1 (bear), 0 (none).
    """
    if not kl or len(kl) < max(n, minbars) + 1:
        return 0
    highs = [float(x[2]) for x in kl]
    lows  = [float(x[3]) for x in kl]
    idx_last = len(kl) - 1

    # find highest high index in last n bars (excluding last bar itself for stability)
    window = highs[idx_last - n: idx_last]
    if not window:
        return 0
    hh = max(window); ii = window.index(hh) + (idx_last - n)
    # find lowest low index in last n bars
    window2 = lows[idx_last - n: idx_last]
    ll = min(window2); jj = window2.index(ll) + (idx_last - n)

    # if last bar broke above the previous local maximum formed at least 'minbars' bars ago -> +1
    if ii <= idx_last - minbars and highs[idx_last] >= hh:
        return +1
    # if last bar broke below the previous local minimum formed at least 'minbars' bars ago -> -1
    if jj <= idx_last - minbars and lows[idx_last] <= ll:
        return -1
    return 0

# ---------------------- Trader ----------------------
class Trader:
    def __init__(self, client: HTTP, symbol: str, inst_cfg: Dict[str, float], ctx: "MultiBot"):
        self.client = client
        self.symbol = symbol
        self.ctx = ctx
        self.qty_step = inst_cfg.get("qty_step", 0.001)
        self.min_qty  = inst_cfg.get("min_qty", 0.001)
        self.tick     = inst_cfg.get("tick", 0.001)

        self.dir = 0      # +1 long, -1 short, 0 flat
        self.qty = 0.0
        self.avg = None   # average entry
        self.legs = 0

        self.entry_fees = 0.0
        self.funding_paid = 0.0
        self.funding_anchor = None
        self.tp_pending = False
        self.tp_px = None
        self.tp_deadline = 0.0
        self.last_bar_ts = None

        # TP maker-first state
        self.tp_pending = False
        self.tp_px = None
        self.tp_deadline = 0.0

    # ------- API helpers -------
    def _switch_one_way(self):
        try:
            rl_switch.wait()
            self.client.switch_position_mode(category=CATEGORY, symbol=self.symbol, mode=0)
            logging.info("[%s] position mode set One-Way", self.symbol)
        except InvalidRequestError as e:
            # 110025: "Position mode is not modified" => OK
            if "110025" in str(e):
                logging.debug("[%s] position mode already One-Way", self.symbol)
            else:
                logging.warning("[%s] switch mode failed: %s", self.symbol, e)
        except Exception as e:
            logging.warning("[%s] switch mode failed: %s", self.symbol, e)

    def _set_leverage(self):
        try:
            rl_lev.wait()
            self.client.set_leverage(category=CATEGORY, symbol=self.symbol,
                                     buyLeverage=str(LEVERAGE_X), sellLeverage=str(LEVERAGE_X))
            logging.info("[%s] leverage set to %.1fx", self.symbol, LEVERAGE_X)
        except InvalidRequestError as e:
            if "110074" in str(e):  # closed contract
                logging.error("[%s] closed contract; disabling trader", self.symbol)
                self.dir = 0
                self.qty = 0.0
                self.avg = None
                self.legs = 0
                # mark as disabled
                setattr(self, "_disabled", True)
            else:
                logging.warning("[%s] set leverage failed: %s", self.symbol, e)
        except Exception as e:
            logging.warning("[%s] set leverage failed: %s", self.symbol, e)

    def ticker(self) -> Tuple[float, float, float]:
        rl_misc.wait()
        r = self.client.get_tickers(category=CATEGORY, symbol=self.symbol)
        tick = float(r["result"]["list"][0]["lastPrice"])
        # Approximate bid/ask around last
        return (tick, tick, tick)

    def klines_1h(self, limit: int = 200) -> List[List[float]]:
        rl_kline.wait()
        r = self.client.get_kline(category=CATEGORY, symbol=self.symbol, interval="60", limit=limit)
        # Normalize [ts, open, high, low, close, volume, ...] → use [ts, open, high, low, close]
        out = []
        for it in r["result"]["list"]:
            ts = int(it[0])
            o = float(it[1]); h = float(it[2]); l = float(it[3]); c = float(it[4])
            out.append([ts, o, h, l, c])
        out.sort(key=lambda x: x[0])
        return out

    # ------- Position ops -------
    def _calc_leg_qty(self, price: float, dir_sign: int) -> float:
        idx = self.legs  # 0-based
        scale = VOL_SCALE_LONG if dir_sign > 0 else VOL_SCALE_SHORT
        leg_usdt = LEG_USDT * (scale ** idx)

        # Cross budget cap
        if CROSS_BUDGET_USDT > 0.0:
            used = self.ctx.total_exposure_usdt(self.symbol, price_hint=price)
            left = CROSS_BUDGET_USDT * (1.0 - RESERVE_PCT) - used
            if left <= 0:
                return 0.0
            leg_usdt = min(leg_usdt, left)

        qty = leg_usdt / price
        qty = max(self.min_qty, round_step(qty, self.qty_step))
        return qty

    def place_market(self, side: str, qty: float, reduce_only: bool = False):

    def cancel_all_orders(self):
        try:
            rl_misc.wait()
            self.client.cancel_all_orders(category=CATEGORY, symbol=self.symbol)
        except Exception as e:
            logging.warning("[%s] cancel_all_orders failed: %s", self.symbol, e)

    def place_limit(self, side: str, qty: float, price: float, reduce_only: bool = False, post_only: bool = False):
        if qty <= 0: return None
        rl_misc.wait()
        tif = "PostOnly" if post_only else "GoodTillCancel"
        try:
            r = self.client.place_order(
                category=CATEGORY, symbol=self.symbol, side=side, orderType="Limit",
                qty=str(qty), price=str(price), reduceOnly=reduce_only, timeInForce=tif
            )
            return r.get("result", {}).get("orderId")
        except Exception as e:
            logging.warning("[%s] place_limit failed: %s", self.symbol, e)
            return None
        if qty <= 0: return
        rl_misc.wait()
        self.client.place_order(
            category=CATEGORY, symbol=self.symbol, side=side, orderType="Market",
            qty=str(qty), reduceOnly=reduce_only, timeInForce="IOC"
        )

    def open_leg(self, dir_sign: int, price: float):
        qty = self._calc_leg_qty(price, dir_sign)
        if qty <= 0:
            logging.info("[%s] skip open leg (qty=0 due to budget/min_qty)", self.symbol)
            return
        side = "Buy" if dir_sign > 0 else "Sell"
        self.place_market(side, qty, reduce_only=False)

        # Update local position
        if self.qty <= 0:
            self.dir = dir_sign
            self.avg = price
            self.qty = qty
            self.legs = 1
        else:
            # merge into avg
            new_qty = self.qty + qty
            self.avg = (self.avg * self.qty + price * qty) / new_qty
            self.qty = new_qty
            self.legs += 1

        # fees/funding tracking
        self.entry_fees += self.qty * price * TAKER_FEE  # approximate
        if self.funding_anchor is None:
            self.funding_anchor = time.time()

        self.tp_pending = False; self.tp_px = None; self.tp_deadline = 0.0
        logging.info("[%s] OPEN leg #%d dir=%+d qty=%.6f @ %.6f (avg=%.6f)",
                     self.symbol, self.legs, self.dir, qty, price, self.avg)

    def close_all(self, price: float):
        if self.dir == 0 or self.qty <= 0: return
        side = "Sell" if self.dir > 0 else "Buy"
        self.place_market(side, self.qty, reduce_only=True)
        logging.info("[%s] CLOSE ALL qty=%.6f @ %.6f (avg=%.6f)", self.symbol, self.qty, price, self.avg)
        # reset
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
        # 1) Bar-close guard
        kl = self.klines_1h(limit=200)
        if not kl or len(kl) < 60:
            logging.debug("[%s] insufficient klines; skip", self.symbol)
            return
        last_ts = int(kl[-1][0])
        if self.last_bar_ts == last_ts:
            return
        self.last_bar_ts = last_ts

        # 2) Signal & price
        sig = lelec_signal_from_klines(kl, n=50, minbars=30)
        _, _, price = self.ticker()

        # 3) Funding accrual
        self._funding_tick(price)

        # 4) TP/SL
        # -- TP: ADA-style maker-first --
        if self.dir != 0 and self.qty > 0 and self.avg is not None and TP_PCT > 0.0:
            want_tp = self.avg * (1.0 + self.dir * TP_PCT)
            # If we are not already pending TP and price reached TP level, place PostOnly reduce-only limit
            if not self.tp_pending:
                reached = (self.dir > 0 and price >= want_tp) or (self.dir < 0 and price <= want_tp)
                if reached:
                    px = round_step(want_tp, self.tick)
                    side = "Sell" if self.dir > 0 else "Buy"
                    oid = self.place_limit(side, self.qty, px, reduce_only=True, post_only=TP_POST_ONLY)
                    if oid:
                        self.tp_pending = True
                        self.tp_px = px
                        self.tp_deadline = time.time() + TP_TTL_SEC
                        logging.info("[%s] TP limit placed (maker-first) qty=%.6f @ %.6f (TTL %.1fs)", self.symbol, self.qty, px, TP_TTL_SEC)
                    else:
                        # fallback immediately if cannot place limit
                        self.close_all(price)
                        return
            else:
                # pending TP: check TTL or price retreat
                if time.time() >= self.tp_deadline:
                    # cancel all and market out
                    self.cancel_all_orders()
                    self.close_all(price)
                    self.tp_pending = False
                    self.tp_px = None
                    self.tp_deadline = 0.0
                    return
                else:
                    # if price retreats away from TP level, you can optionally cancel early
                    pass

        # -- SL: emergency market (fast exit) --
        if self.dir != 0 and self.qty > 0 and self.avg is not None and EMERGENCY_SL_PCT > 0.0:
            sl_px = self.avg * (1.0 - self.dir * EMERGENCY_SL_PCT)
            hit = (self.dir > 0 and price <= sl_px) or (self.dir < 0 and price >= sl_px)
            if hit:
                # Cancel TP limit if any to free position size, then market out
                if self.tp_pending:
                    self.cancel_all_orders()
                    self.tp_pending = False
                    self.tp_px = None
                    self.tp_deadline = 0.0
                self.close_all(price)
                return

        # 5) Flat → open if signal
        if self.dir == 0:
            if sig != 0:
                self.open_leg(sig, price)
            return

        # 6) In position:
        upnl = self.unreal_pnl(price)

        if sig == self.dir:
            # same-side → DCA if losing
            if upnl < 0.0:
                # MAX_DCA: unlimited if <0
                if MAX_DCA < 0 or self.legs < 1 + MAX_DCA:
                    self.open_leg(self.dir, price)
                else:
                    logging.info("[%s] MAX_DCA reached; skip DCA", self.symbol)
            else:
                # breakeven/no-signal exit handled below
                pass

        elif sig == -self.dir:
            # opposite signal
            if self.legs == 1:
                # auto-reverse same bar
                self.close_all(price)
                self.open_leg(-self.dir, price)
                return
            else:
                # require breakeven OR (optionally) profit
                if upnl >= 0.0 or (FLIP_ON_PROFIT and upnl > 0.0):
                    self.close_all(price)
                    self.open_leg(-self.dir, price)
                    return

        # 7) No signal or unhandled cases → if DCA'd and breakeven, close
        if self.legs > 1 and upnl >= 0.0 and sig == 0:
            self.close_all(price)

    # Helpers for initial sync (mode/leverage) can be called by MultiBot
    def init_on_exchange(self):
        if getattr(self, "_disabled", False):
            return
        self._switch_one_way()
        self._set_leverage()

# ---------------------- MultiBot ----------------------
class MultiBot:
    def __init__(self, symbols: List[str]):
        self.client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET, recv_window=5000)
        self.symbols = self._filter_live(symbols)
        self.traders: Dict[str, Trader] = {}
        inst_map = self._load_instruments()

        for s in self.symbols:
            cfg = inst_map.get(s, {"qty_step": 0.001, "min_qty": 0.001, "tick": 0.001})
            t = Trader(self.client, s, cfg, ctx=self)
            t.init_on_exchange()
            self.traders[s] = t

        logging.info("MultiBot started for %d symbols; LEG_USDT=%.4f, lev=%.1fx, fee=%.4f, funding_8h=%.6f, order_mode=%s, max_dca=%d",
                     len(self.traders), LEG_USDT, LEVERAGE_X, TAKER_FEE, FUNDING_8H, ORDER_MODE, MAX_DCA)

    def _load_instruments(self) -> Dict[str, Dict[str, float]]:
        # Build lot size & tick mapping
        m: Dict[str, Dict[str, float]] = {}
        rl_misc.wait()
        r = self.client.get_instruments_info(category=CATEGORY)
        for it in r["result"]["list"]:
            sym = it["symbol"].upper()
            lot = it.get("lotSizeFilter", {}) or {}
            px  = it.get("priceFilter", {}) or {}
            qty_step = float(lot.get("qtyStep", "0.001"))
            min_qty  = float(lot.get("minOrderQty", "0.001"))
            tick     = float(px.get("tickSize", "0.001"))
            m[sym] = {"qty_step": qty_step, "min_qty": min_qty, "tick": tick, "status": it.get("status", "")}
        return m

    def _filter_live(self, symbols: List[str]) -> List[str]:
        # Pre-filter by instruments info status
        live = []
        info = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET, recv_window=5000).get_instruments_info(category=CATEGORY)
        status_map = {it["symbol"].upper(): it.get("status", "") for it in info["result"]["list"]}
        for s in symbols:
            st = str(status_map.get(s, "")).lower()
            if st in ("trading", "1", "live"):
                live.append(s)
            else:
                logging.error("[%s] contract not live (%s); skip", s, st)
        return live

    def total_exposure_usdt(self, sym_hint: Optional[str] = None, price_hint: Optional[float] = None) -> float:
        # Sum of |qty * price| across all traders; use last ticker for each
        tot = 0.0
        for s, t in self.traders.items():
            if t.dir == 0 or t.qty <= 0: 
                continue
            if s == sym_hint and price_hint is not None:
                price = price_hint
            else:
                try:
                    _,_, price = t.ticker()
                except Exception:
                    price = t.avg or 0.0
            tot += abs(t.qty * price)
        return tot

    def loop(self):
        while True:
            for s, t in list(self.traders.items()):
                if getattr(t, "_disabled", False):
                    continue
                try:
                    t.step()
                except Exception as e:
                    logging.exception("[%s] step error: %s", s, e)
            time.sleep(POLL_SEC)

# ---------------------- Main ----------------------
if __name__ == "__main__":
    syms = read_pairs(PAIRS_FILE)
    if not syms:
        raise SystemExit(f"No symbols loaded from {PAIRS_FILE}")
    bot = MultiBot(syms)
    bot.loop()
