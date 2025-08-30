
# 46cap Bulls DCA — Multi-pair (Render + GitHub)

Chạy **đa cặp** theo `46cap.txt` — không có bản đơn cặp để tránh nhầm. Logic **auto-reverse như 18.py**.

## Chiến lược
- Tín hiệu Lelec 1H (`length=50`, `bars=30`) tại **đóng nến**.
- Vào lệnh **2.5 USDT/leg**, đòn bẩy 5× (set trên sàn).
- Tín hiệu **cùng chiều** & **đang lỗ** → **DCA +2.5 USDT**.
- Tín hiệu **ngược**:
  - **legs==1** → **đóng & mở ngược ngay** trên **cùng bar** (auto-reverse).
  - **legs>1** → **đóng & mở ngược** **chỉ khi** **PnL ≥ 0**; nếu **PnL < 0** → **bỏ qua** tín hiệu.
- **Không tín hiệu**: nếu **legs>1** & **PnL ≥ 0** → **đóng** (breakeven) và **đứng ngoài**.
- Để so sánh “chỉ theo tín hiệu”, mặc định `TAKER_FEE=0`, `FUNDING_RATE_8H=0` (đổi lại khi chạy thực).

## Deploy (Render)
1. Push repo này lên GitHub.
2. Render → New → **Background Worker** → chọn repo.
3. Blueprint: **render.yaml** (start `python bot_multi.py`).
4. Env cần thiết: `BYBIT_API_KEY`, `BYBIT_API_SECRET`.
5. Tuỳ chọn: `PAIRS_FILE`, `LEG_USDT`, `LEVERAGE_X`, `TAKER_FEE`, `FUNDING_RATE_8H`, `ORDER_MODE`, `MAKER_TTL_SEC`, `MAKER_OFFSET_BPS`, `POLL_SEC`.

## 46cap.txt
- Mỗi dòng 1 symbol, `USDT` có thể bỏ (code sẽ tự thêm). Ví dụ:
```
ADAUSDT
ETHUSDT
BTCUSDT
...
```

## Lưu ý
- Bot giữ state **trong RAM**; restart sẽ sync size/avg từ sàn nhưng **không biết số legs** trước đó → giả định `legs=1`. Cần chính xác → dùng Redis/DB.
- `ORDER_MODE=maker_then_market` giúp giảm phí nhưng vẫn đảm bảo khớp nếu không fill kịp.
