
# bot_multi.py — Multi-symbol Bulls DCA bot (Render) — auto-reverse like 18.py
# Strategy (bar-close, Lelec 50/30, 1H):
# - Flat + signal => open 1 leg (LEG_USDT) in signal direction.
# - Same-side signal while losing (PnL<0) => DCA +LEG_USDT.
# - Opposite signal:
#     * if legs==1: CLOSE then immediately OPEN opposite (auto-reverse on same bar).
#     * if legs>1: CLOSE & auto-reverse only if breakeven (PnL>=0); else ignore signal.
# - No signal: if legs>1 and breakeven => CLOSE (stay flat).
# PnL check includes entry fees (per leg) + exit fee + funding (if enabled). For “signal parity” use TAKER_FEE=0, FUNDING_RATE_8H=0.
import os, time, math, uuid, logging
from typing import List, Dict
from pybit.unified_trading import HTTP

def env_float(name, default): 
    try: return float(os.environ.get(name, default))
    except: return default

LOG_LEVEL  = os.environ.get("LOG_LEVEL","INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s: %(message)s")

API_KEY = os.environ.get("BYBIT_API_KEY") or os.environ.get("API_KEY")
API_SEC = os.environ.get("BYBIT_API_SECRET") or os.environ.get("API_SECRET")
if not API_KEY or not API_SEC:
    raise SystemExit("Set BYBIT_API_KEY and BYBIT_API_SECRET")

CATEGORY = os.environ.get("CATEGORY","linear")
POLL_SEC = env_float("POLL_SEC", 30.0)

LEG_USDT   = env_float("LEG_USDT", 2.5)
LEVERAGE_X = env_float("LEVERAGE_X", 5.0)
TAKER_FEE  = env_float("TAKER_FEE", 0.0)    # default 0 for parity with backtest timing
FUNDING_8H = env_float("FUNDING_RATE_8H", 0.0)

ORDER_MODE       = os.environ.get("ORDER_MODE","maker_then_market")  # market|maker|maker_then_market
MAKER_TTL_SEC    = env_float("MAKER_TTL_SEC", 10.0)
MAKER_OFFSET_BPS = env_float("MAKER_OFFSET_BPS", 1.0)  # 1 = 0.01%

PAIRS_FILE = os.environ.get("PAIRS_FILE","46cap.txt")

def read_pairs(path: str) -> list:
    syms = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"): continue
                s = s.upper()
                if not s.endswith("USDT"): s += "USDT"
                syms.append(s)
    except Exception as e:
        logging.error("Cannot read PAIRS_FILE=%s: %s", path, e)
    return sorted(list(dict.fromkeys(syms)))

def bulls_signal_from_klines(klines: List[List[str]]):
    length = 50; bars = 30
    o = [float(x[1]) for x in klines]
    h = [float(x[2]) for x in klines]
    l = [float(x[3]) for x in klines]
    c = [float(x[4]) for x in klines]
    n = len(c)
    if n < max(length, 35): return 0
    highest = [max(h[max(0, i-length+1):i+1]) for i in range(n)]
    lowest  = [min(l[max(0, i-length+1):i+1]) for i in range(n)]
    bindex = [0]*n; sindex = [0]*n; lelex = [0]*n
    for i in range(n):
        if i>=1: bindex[i], sindex[i] = bindex[i-1], sindex[i-1]
        if i>=4 and c[i] > c[i-4]: bindex[i]+=1
        if i>=4 and c[i] < c[i-4]: sindex[i]+=1
        condShort = bindex[i] > bars and c[i] < o[i] and h[i] >= highest[i]
        condLong  = sindex[i] > bars and c[i] > o[i] and l[i] <= lowest[i]
        if condShort: bindex[i] = 0; lelex[i] = -1
        elif condLong: sindex[i] = 0; lelex[i] = 1
    return lelex[-1]

class Trader:
    def __init__(self, http: HTTP, symbol: str):
        self.http = http; self.symbol = symbol
        info = self.http.get_instruments_info(category=CATEGORY, symbol=symbol)
        inst = info["result"]["list"][0]
        lot = inst["lotSizeFilter"]; pricef = inst["priceFilter"]
        self.qty_step = float(lot["qtyStep"]); self.min_qty = float(lot["minOrderQty"])
        self.tick_size = float(pricef["tickSize"])

        try:
            self.http.switch_position_mode(category=CATEGORY, symbol=symbol, mode=0)
        except Exception as e:
            logging.warning("[%s] position mode set failed: %s", symbol, e)
        try:
            self.http.set_leverage(category=CATEGORY, symbol=symbol,
                                   buyLeverage=str(LEVERAGE_X), sellLeverage=str(LEVERAGE_X))
        except Exception as e:
            logging.warning("[%s] set leverage failed: %s", symbol, e)

        self.dir = 0; self.qty=0.0; self.avg=None; self.legs=0
        self.entry_fees = 0.0; self.funding_paid=0.0; self.funding_anchor=None
        self.last_bar_ts = None
        self.sync_from_exchange()

    def _rq(self, q):
        q = math.floor(q / self.qty_step) * self.qty_step
        if 0 < q < self.min_qty: q = self.min_qty
        return q
    def _rp(self, p): return math.floor(p / self.tick_size) * self.tick_size

    def ticker(self):
        r = self.http.get_tickers(category=CATEGORY, symbol=self.symbol)
        it = r["result"]["list"][0]
        last = float(it["lastPrice"])
        bid  = float(it.get("bid1Price") or last)
        ask  = float(it.get("ask1Price") or last)
        mid = (bid+ask)/2.0
        return bid, ask, mid

    def klines_1h(self, limit=200):
        r = self.http.get_kline(category=CATEGORY, symbol=self.symbol, interval="60", limit=limit)
        return sorted(r["result"]["list"], key=lambda x: int(x[0]))

    def sync_from_exchange(self):
        try:
            r = self.http.get_positions(category=CATEGORY, symbol=self.symbol)
            lst = r.get("result",{}).get("list",[])
            size = 0.0; avg = None; side = 0; p = None
            for p in lst:
                sz = float(p.get("size") or 0.0)
                if sz > 0:
                    size = sz
                    side = +1 if p.get("side","")=="Buy" or p.get("positionIdx") in (1,0) else -1
                    avg = float(p.get("avgPrice") or 0.0)
                    break
            if size > 0:
                self.dir = side; self.qty = size; self.avg = avg; self.legs = 1
                self.entry_fees = size*avg*TAKER_FEE; self.funding_paid=0.0; self.funding_anchor=time.time()
            else:
                self.dir=0; self.qty=0.0; self.avg=None; self.legs=0; self.entry_fees=0.0; self.funding_paid=0.0; self.funding_anchor=None
        except Exception as e:
            logging.warning("[%s] sync failed: %s", self.symbol, e)

    def place_with_mode(self, side: str, qty: float, reduce: bool):
        qty = self._rq(qty)
        if qty <= 0: return False
        if ORDER_MODE == "market":
            self.http.place_order(category=CATEGORY, symbol=self.symbol, side=("Buy" if side=="long" else "Sell"),
                                  orderType="Market", qty=str(qty), reduceOnly=reduce)
            return True
        # maker or maker_then_market
        bid, ask, mid = self.ticker()
        if side == "long":
            px = min(bid, mid * (1.0 - MAKER_OFFSET_BPS/10000.0))
        else:
            px = max(ask, mid * (1.0 + MAKER_OFFSET_BPS/10000.0))
        px = self._rp(px)
        link = str(uuid.uuid4())
        try:
            self.http.place_order(category=CATEGORY, symbol=self.symbol, side=("Buy" if side=="long" else "Sell"),
                                  orderType="Limit", qty=str(qty), price=str(px),
                                  timeInForce="PostOnly", reduceOnly=reduce, orderLinkId=link)
        except Exception as e:
            if ORDER_MODE == "maker_then_market":
                self.http.place_order(category=CATEGORY, symbol=self.symbol, side=("Buy" if side=="long" else "Sell"),
                                      orderType="Market", qty=str(qty), reduceOnly=reduce)
                return True
            return False
        # wait
        t0 = time.time()
        while time.time() - t0 < MAKER_TTL_SEC:
            time.sleep(1.0)
            q = self.http.get_open_orders(category=CATEGORY, symbol=self.symbol, orderLinkId=link)
            if len(q.get("result",{}).get("list",[])) == 0:
                return True
        try:
            self.http.cancel_order(category=CATEGORY, symbol=self.symbol, orderLinkId=link)
        except: pass
        if ORDER_MODE == "maker_then_market":
            self.http.place_order(category=CATEGORY, symbol=self.symbol, side=("Buy" if side=="long" else "Sell"),
                                  orderType="Market", qty=str(qty), reduceOnly=reduce)
            return True
        return False

    def open_leg(self, direction: int, ref_px: float):
        qty = self._rq(LEG_USDT / max(ref_px,1e-9))
        if qty <= 0:
            logging.warning("[%s] Qty too small; increase LEG_USDT.", self.symbol)
            return False
        ok = self.place_with_mode("long" if direction>0 else "short", qty, reduce=False)
        if not ok: return False
        # assume fill at mid
        _,_, mid = self.ticker()
        fill_px = mid
        if self.dir == 0:
            self.dir = direction; self.qty = qty; self.avg = fill_px; self.legs = 1
            self.entry_fees = qty*fill_px*TAKER_FEE; self.funding_paid=0.0; self.funding_anchor=time.time()
        else:
            new_qty = self.qty + qty
            self.avg = ((self.avg*self.qty)+fill_px*qty)/max(new_qty,1e-9)
            self.qty = new_qty; self.legs += 1; self.entry_fees += qty*fill_px*TAKER_FEE
        return True

    def close_all(self):
        if self.dir == 0 or self.qty<=0: return
        side = "long" if self.dir>0 else "short"
        self.place_with_mode(side, self.qty, reduce=True)
        self.dir=0; self.qty=0.0; self.avg=None; self.legs=0; self.entry_fees=0.0; self.funding_paid=0.0; self.funding_anchor=None

    def accrue_funding(self, price: float):
        if FUNDING_8H <= 0 or self.dir == 0 or self.qty <= 0 or self.funding_anchor is None:
            return
        now = time.time(); period = 8*3600.0
        while self.funding_anchor + period <= now:
            notional = self.qty * price
            self.funding_paid += notional * FUNDING_8H
            self.funding_anchor += period

    def unreal_pnl(self, price: float) -> float:
        if self.dir == 0 or self.avg is None or self.qty<=0: return 0.0
        gross = (price/self.avg - 1.0) if self.dir>0 else (1.0 - price/self.avg)
        px_pnl = self.qty * self.avg * gross
        exit_fee = self.qty * price * TAKER_FEE
        return px_pnl - (self.entry_fees + exit_fee + self.funding_paid)

    def step(self):
        kl = self.klines_1h(limit=200)
        last_ts = int(kl[-1][0])
        if getattr(self, "last_bar_ts", None) == last_ts:
            return
        self.last_bar_ts = last_ts

        sig = bulls_signal_from_klines(kl)
        _,_, mid = self.ticker()
        price = mid
        self.accrue_funding(price)
        upnl = self.unreal_pnl(price)

        if self.dir == 0:
            if sig != 0:
                self.open_leg(sig, price)
            return

        if sig == self.dir:
            if upnl < 0:
                self.open_leg(self.dir, price)
            return

        if sig == -self.dir:
            if self.legs <= 1:
                self.close_all()
                self.open_leg(sig, price)
            else:
                if upnl >= 0:
                    self.close_all()
                    self.open_leg(sig, price)
            return

        if self.legs > 1 and upnl >= 0:
            self.close_all()

class MultiBot:
    def __init__(self, symbols: list):
        self.http = HTTP(api_key=API_KEY, api_secret=API_SEC, recv_window=60000)
        self.traders: Dict[str, Trader] = {}
        for s in symbols:
            try:
                self.traders[s] = Trader(self.http, s)
                logging.info("Loaded %s", s)
            except Exception as e:
                logging.error("Init %s failed: %s", s, e)

    def loop(self):
        logging.info("MultiBot started for %d symbols; LEG_USDT=%.4f, lev=%sx, fee=%.4f, funding_8h=%.6f, order_mode=%s",
                     len(self.traders), LEG_USDT, LEVERAGE_X, TAKER_FEE, FUNDING_8H, ORDER_MODE)
        while True:
            for s, t in list(self.traders.items()):
                try:
                    t.step()
                except Exception as e:
                    logging.exception("[%s] step error: %s", s, e)
            time.sleep(POLL_SEC)

if __name__ == "__main__":
    pairs = read_pairs(PAIRS_FILE)
    if not pairs:
        raise SystemExit(f"No symbols loaded from {PAIRS_FILE}")
    MultiBot(pairs).loop()
