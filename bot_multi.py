# bot_multi.py — Multi-symbol DCA bot for Bybit linear perp
# Core rules (parity with 18.py, breakeven ignores fees):
# - On each closed 1H bar: compute Lelec(50/30) signal s ∈ {-1,0,+1}.
# - If Flat and s!=0: OPEN 1 leg in direction s.
# - If In Position and s==dir and unrealized PnL<0: DCA one more leg.
# - If Opposite signal:
#     * if legs==1: CLOSE then immediately OPEN opposite (same bar).
#     * if legs>1: only close&reverse when breakeven (PnL>=0).
# - If No signal and legs>1 and breakeven: CLOSE now.
#
# Extras:
# - MAX_DCA default -1 (unlimited). Flip-on-profit switch.
# - Geo-scaling per leg (VOL_SCALE_LONG/SHORT).
# - Cross-budget cap across all symbols.
# - TP: maker-first (PostOnly) with TTL; fallback to market. SL: emergency market.
# - Throttled API calls; filter non-live with kline fallback.
# - Fail-fast if API keys missing.
#
import os, time, logging
from typing import List, Dict, Optional, Tuple
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError

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
PAIRS_FILE       = env_str("PAIRS_FILE", "46cap.txt")
POLL_SEC         = env_float("POLL_SEC", 2.5)

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise SystemExit("BYBIT_API_KEY / BYBIT_API_SECRET are required but missing. Set them in Render → Environment.")

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

def round_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    k = round(x / step)
    v = k * step
    return v if v > 0 else step

def lelec_signal_from_klines(kl: List[List[float]], n: int = 50, minbars: int = 30) -> int:
    if not kl or len(kl) < max(n, minbars) + 1:
        return 0
    highs = [float(x[2]) for x in kl]
    lows  = [float(x[3]) for x in kl]
    last  = len(kl) - 1
    seg_hi = highs[last - n:last]
    seg_lo = lows[last - n:last]
    if not seg_hi or not seg_lo:
        return 0
    hh = max(seg_hi); hi = seg_hi.index(hh) + (last - n)
    ll = min(seg_lo);  li = seg_lo.index(ll) + (last - n)
    if hi <= last - minbars and highs[last] >= hh:
        return +1
    if li <= last - minbars and lows[last]  <= ll:
        return -1
    return 0

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
            logging.info("[%s] position mode set One-Way", self.symbol)
        except InvalidRequestError as e:
            if "110025" in str(e):  # already one-way
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
            logging.info("[%s] leverage set to %.1fx", self.symbol, LEVERAGE_X)
        except InvalidRequestError as e:
            if "110074" in str(e):
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
        qty = round_step(qty, self.qty_step)
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
            logging.warning("[%s] cancel_all_orders failed: %s", self.symbol, e)

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

        self.entry_fees += self.qty * price * TAKER_FEE
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
        if not kl or len(kl) < 60:
            logging.debug("[%s] insufficient klines; skip", self.symbol)
            return
        last_ts = int(kl[-1][0])
        if self.last_bar_ts == last_ts:
            return
        self.last_bar_ts = last_ts

        sig = lelec_signal_from_klines(kl, n=50, minbars=30)
        _, _, price = self.ticker()

        self._funding_tick(price)

        # TP maker-first with TTL; SL market
        if self.dir != 0 and self.qty > 0 and self.avg is not None:
            if TP_PCT > 0.0:
                want_tp = self.avg * (1.0 + self.dir * TP_PCT)
                if not self.tp_pending:
                    reached = (self.dir > 0 and price >= want_tp) or (self.dir < 0 and price <= want_tp)
                    if reached:
                        tp_px = round_step(want_tp, self.tick)
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
        logging.info("MultiBot started for %d symbols; LEG_USDT=%.4f, lev=%.1fx, fee=%.4f, funding_8h=%.6f",
                     len(self.traders), LEG_USDT, LEVERAGE_X, TAKER_FEE, FUNDING_8H)

    def _load_instruments(self) -> Dict[str, Dict[str, float]]:
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

                logging.error("[%s] contract not live (%s); skip", s, st or "unknown")
            except Exception as e:
                # API hiccup -> allow symbol; runtime will handle errors later
                logging.warning("[%s] status check error: %s; allowing symbol", s, e)
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

if __name__ == "__main__":
    syms = read_pairs(PAIRS_FILE)
    if not syms:
        raise SystemExit(f"No symbols loaded from {PAIRS_FILE}")
    MultiBot(syms).loop()
