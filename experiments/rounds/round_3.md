# ABGT 優化迴圈 — 第 3 輪回合報告

日期：2026-09-23　｜　基準：`ema50_plain`（六格平均 Sharpe 1.09、十格全過、holdout SOL 5/5 / BNB 3/5）　｜　本輪變體：10 個（5 個視角 × 2）　｜　實作失敗：0　｜　驗證：前 3 名各做 lookahead / overfit / executable / holdout 四道

## 0. 結論一句話

**無新冠軍，基準維持 `ema50_plain`；這是連續第 3 輪，依 CLAUDE.md 規則迴圈停止。** 十個變體有八個十格全過、八個六格平均 Sharpe 高於基準（最高 `slow_vol_scaled_band_atr50` 1.22），三個同時滿足「十格全過 + 六格平均 > 1.09 + 四 OOS Δ ≥ −0.05」的數值門檻（`volume_confirmed_entry_q1_ema_exit`、`btc_leader_gate_ema50_and`、`volume_confirmed_entry_q1_band_exit`），是三輪以來第一次有變體走到驗證階段時數值門檻全部達標；但三個的 overfit 與 executable 審查全部判 fail，理由一致：**六格平均的全部增量都在 IS（2020–2022 熊市反彈的鋸齒成本），四個 OOS 格對冠軍的 Δ 在 ±0.03 內、|t| < 0.4，扣掉前 2–3 個事件日就歸零**。可執行審查者做的安慰劑檢定給出本輪最重要的方法論結論：把量能比值隨機打亂 150 次，**25% 的隨機進場濾網同樣滿足全部取代門檻**——現行「六格平均高於冠軍」的門檻對「疊在 EMA50 上的任何進場減速器」沒有鑑別力，因為 FULL 含 IS 而 IS 的鋸齒成本任何濾網都拿得到。

第 2 輪建議的「改資訊集（跨標的、成交量）或部位表示（分級曝險）」本輪執行完畢，結果：成交量在 BTC 有超出價格距離的資訊（等拒絕率距離濾網消融 FULL t 2.04）但 OOS 為零、在 ETH 只是距離代理；BTC 領先閘的 ETH 增益是 2020-03-08 與 2021-05-19 兩天、OOS 四種機制互相抵銷；分級曝險把「熊市反彈押半倉賺、V 型反轉押半倉賠」的 IS/OOS 分裂原樣搬到倉位維度。三輪 32 個變體、320 個對基準的格，本輪出現了頭兩個 |t| ≥ 2 的正向格（BTC IS：`context_adaptive_entry_exit_cooldown` 2.31、`volume_confirmed_entry_q1_ema_exit` 2.12），全部在 IS；OOS 上 |t| ≥ 2 的仍只有負差異（第 2 輪 relvol_downside、本輪階梯消融 B2 t −2.98/−2.54）。

**建議「停止」的內容**：`ema50_plain` 做 Pine 版與現貨紙上前向、把引擎資源轉到運行閘門（第 11 節 P1）與雙標的組合（P4），而不是第 4 輪 EMA50 濾波變體。若使用者決定例外續跑，第 12 節列出唯一沒被本輪否定的線索（分級曝險的純 EMA50 出場版 A3、分級出場、熊市情境門檻）與其事前判準。

---

## 1. 評估設定

- 標的：BTC/USDT:USDT、ETH/USDT:USDT 日線（Binance USDT-M 永續）；holdout SOL/USDT:USDT（2020-09 起）、BNB/USDT:USDT（2020-02 起）
- 十格：IS 2020-01→2022-12、OOS 2023-01→2026-09-22、FULL；替換切分 IS2 2020-07→2023-12、OOS2 2024-01→今；× BTC/ETH
- 引擎：`regime.py` v3（第 2 輪 P1–P4 已實作：`exiting` 狀態、`scale` 分級曝險、`exposure_key`/`fill_key`/`n_fills`、逐日歷史資金費率），只做多、收盤判定、下一根開盤成交、0.05%/邊、0.02% 滑點
- 閘門：Sharpe > 同期持有 且 Calmar > 持有 且 最大回撤 < 持有，每格各自判定
- 同期持有 Sharpe：BTC IS 0.76 / OOS 1.18 / FULL 0.92 / IS2 1.00 / OOS2 0.75；ETH IS 1.18 / OOS 0.665 / FULL 0.93 / IS2 1.20 / OOS2 0.42
- 敏感度：±25%（regime 與曝險都相同的擾動去重）；壓力：費 0.10%+滑 0.05%、費 0.10%+滑 0.20%、資金費率 0.01%/8h、延遲 1 根、歷史資金費率
- 取代冠軍門檻：十格 ≥ 9 且四個 OOS 格全過、六格平均 Sharpe > 1.09、四個 OOS 格對冠軍 ΔSharpe ≥ −0.05、OOS 敏感度通過率 ≥ 0.67、lookahead / overfit / holdout 三道驗證 pass
- 基準壓力列以 `results/variants/ema50_plain.{json,txt}` 為準：延遲一根 ETH 是 **5/5**（OOS 0.82），第 2 輪報告寫的 5/4 是舊數字；本輪部分實作者沿用了 5/4，讀壓力表時以本報告第 5 節為準

本輪五個視角與對應的第 2 輪路線圖：

| 視角 | 變體 | 第 2 輪 §11 方向 |
|------|------|-----------------|
| context_adaptive_entry | `context_adaptive_entry_exit_cooldown`（b 出場冷卻）、`context_adaptive_entry_whipsaw_count`（a 影子鞭打計數） | 1：進場門檻依情境在 plain 與 channel 之間切換 |
| btc_leader_gate | `btc_leader_gate_ema50_and`（規則 A）、`btc_leader_gate_relstrength_ratio`（規則 B 對照） | 2：用 BTC 趨勢狀態做其他標的的閘門 |
| slow_vol_scaled_band | `slow_vol_scaled_band_median365`（A）、`slow_vol_scaled_band_atr50`（B 對照） | 3：通道帶寬改慢速波動尺度 |
| volume_confirmed_entry | `volume_confirmed_entry_q1_ema_exit`、`volume_confirmed_entry_q1_band_exit` | 4：成交量當進場確認 |
| graded_exposure | `graded_exposure_dist_ratchet`（a 距離分級）、`graded_exposure_ema_ladder_mean`（b 長度分散） | 5：分級曝險（依賴引擎 `scale`） |

---

## 2. 總表：變體 × 十格

排序 = 第 10 節的「距取代門檻的遠近」。格式 Sharpe / 最大回撤，✓ = 過閘門。`vol_atr_channel`（第 1 輪最佳、`atr_channel.py`）列為對照。

### 2.1 BTC 五格

| # | 變體 | IS | OOS | FULL | IS2 | OOS2 |
|---|------|----|-----|------|-----|------|
| — | **ema50_plain（基準）** | 1.01 / −56% ✓ | 1.40 / −30% ✓ | 1.17 / −56% ✓ | 1.32 / −56% ✓ | 0.97 / −30% ✓ |
| 對照 | vol_atr_channel | 1.28 / −46% ✓ | 1.34 / −24% ✓ | 1.29 / −46% ✓ | 1.50 / −46% ✓ | 0.94 / −24% ✓ |
| 1 | volume_confirmed_entry_q1_ema_exit | 1.32 / −47% ✓ | 1.39 / −29% ✓ | 1.33 / −47% ✓ | 1.50 / −47% ✓ | 0.99 / −29% ✓ |
| 2 | btc_leader_gate_ema50_and | 1.01 / −56% ✓ | 1.40 / −30% ✓ | 1.17 / −56% ✓ | 1.32 / −56% ✓ | 0.97 / −30% ✓ |
| 3 | volume_confirmed_entry_q1_band_exit | 1.14 / −54% ✓ | 1.44 / −29% ✓ | 1.26 / −54% ✓ | 1.36 / −54% ✓ | 1.09 / −29% ✓ |
| 4 | slow_vol_scaled_band_atr50 | 1.39 / −37% ✓ | 1.34 / −23% ✓ | 1.35 / −37% ✓ | 1.59 / −37% ✓ | 0.97 / −23% ✓ |
| 5 | slow_vol_scaled_band_median365 | 1.37 / −38% ✓ | 1.24 / −26% ✓ | 1.29 / −38% ✓ | 1.50 / −38% ✓ | 0.93 / −26% ✓ |
| 6 | context_adaptive_entry_exit_cooldown | 1.28 / −49% ✓ | 1.33 / −31% ✓ | 1.29 / −49% ✓ | 1.51 / −49% ✓ | 0.90 / −31% ✓ |
| 7 | graded_exposure_ema_ladder_mean | 1.19 / −48% ✓ | 1.33 / −33% ✓ | 1.23 / −48% ✓ | 1.42 / −48% ✓ | 0.91 / −33% ✓ |
| 8 | graded_exposure_dist_ratchet | 1.08 / −56% ✓ | 1.33 / −30% ✓ | 1.18 / −56% ✓ | 1.33 / −56% ✓ | 0.92 / −30% ✓ |
| 9 | context_adaptive_entry_whipsaw_count | 0.89 / −64% ✓ | 1.27 / −33% ✓ | 1.05 / −64% ✓ | 1.19 / −64% ✓ | 0.80 / −33% ✓ |
| 10 | btc_leader_gate_relstrength_ratio | 1.01 / −56% ✓ | 1.40 / −30% ✓ | 1.17 / −56% ✓ | 1.32 / −56% ✓ | 0.97 / −30% ✓ |

兩個 btc_leader 變體的 BTC 分支直接回傳基準（regime_key 逐位相同），BTC 五格 Δ 恆為 0。

### 2.2 ETH 五格

| # | 變體 | IS | OOS | FULL | IS2 | OOS2 |
|---|------|----|-----|------|-----|------|
| — | **ema50_plain（基準）** | 1.26 / −61% ✓ | 0.71 / −51% ✓ | 0.99 / −61% ✓ | 1.22 / −61% ✓ | 0.45 / −51% ✓ |
| 對照 | vol_atr_channel | 1.43 / −46% ✓ | 0.71 / −48% ✓ | 1.09 / −48% ✓ | 1.32 / −46% ✓ | 0.55 / −48% ✓ |
| 1 | volume_confirmed_entry_q1_ema_exit | 1.38 / −51% ✓ | 0.69 / −54% ✓ | 1.05 / −54% ✓ | 1.33 / −51% ✓ | 0.45 / −54% ✓ |
| 2 | btc_leader_gate_ema50_and | 1.57 / −50% ✓ | 0.73 / −48% ✓ | 1.17 / −50% ✓ | 1.56 / −50% ✓ | 0.43 / −48% ✓ |
| 3 | volume_confirmed_entry_q1_band_exit | 1.40 / −51% ✓ | 0.70 / −51% ✓ | 1.07 / −51% ✓ | 1.35 / −51% ✓ | 0.43 / −51% ✓ |
| 4 | slow_vol_scaled_band_atr50 | 1.43 / −49% ✓ | 0.73 / −51% ✓ | 1.09 / −51% ✓ | 1.31 / −49% ✓ | 0.59 / −51% ✓ |
| 5 | slow_vol_scaled_band_median365 | 1.41 / −52% ✓ | 0.71 / −48% ✓ | 1.07 / −52% ✓ | 1.31 / −52% ✓ | 0.52 / −48% ✓ |
| 6 | context_adaptive_entry_exit_cooldown | 1.27 / −62% ✓ | 0.75 / −54% ✓ | 1.01 / −62% ✓ | 1.25 / −62% ✓ | 0.48 / −54% ✓ |
| 7 | graded_exposure_ema_ladder_mean | 1.28 / −60% ✓ | 0.80 / −48% ✓ | 1.04 / −61% ✓ | 1.25 / −61% ✓ | 0.61 / −48% ✓ |
| 8 | graded_exposure_dist_ratchet | 1.34 / −56% ✓ | 0.75 / −49% ✓ | 1.05 / −56% ✓ | 1.27 / −56% ✓ | 0.54 / −49% ✓ |
| 9 | context_adaptive_entry_whipsaw_count | 1.22 / −67% ✓ | 0.80 / −50% ✓ | 1.01 / −67% ✓ | 1.19 / −67% ✗ | 0.56 / −50% ✓ |
| 10 | btc_leader_gate_relstrength_ratio | 0.64 / −70% ✗ | 0.67 / −40% ✓ | 0.61 / −79% ✗ | 0.30 / −77% ✗ | 1.05 / −37% ✓ |

### 2.3 綜合：閘門數與取代門檻檢查

| # | 變體 | 十格 | 六格平均 Sharpe | 四 OOS 格 Δ 最小 | OOS 敏感度最低 | 取代門檻未過的項目 |
|---|------|------|----------------|-----------------|---------------|-------------------|
| — | ema50_plain | 10/10 | 1.09（重跑 1.092） | — | 1.00 | — |
| 1 | volume_confirmed_entry_q1_ema_exit | 10/10 | 1.192 | −0.023 | 1.00 | 數值門檻全過；**overfit ✗、executable ✗**；D7 事前規則觸發 |
| 2 | btc_leader_gate_ema50_and | 10/10 | 1.175 | −0.022 | **0.50** | OOS 敏感度（btc_ema_len 38/62 ETH OOS/OOS2 四格全掛）；**overfit ✗、executable ✗**；方向專屬判準（ETH OOS/OOS2 Δ ≥ +0.03、任一格 \|t\| ≥ 1）三項全不過 |
| 3 | volume_confirmed_entry_q1_band_exit | 10/10 | 1.168 | −0.026 | 1.00 | 數值門檻全過；**overfit ✗、executable ✗**；holdout 5/10 踩線、BNB 平均 Δ −0.096（分標的規則不過） |
| 4 | slow_vol_scaled_band_atr50 | 10/10 | **1.222** | **−0.057** | 0.75 | BTC OOS Δ（差 0.007）；holdout SOL 3/5、BNB 0/5 |
| 5 | slow_vol_scaled_band_median365 | 10/10 | 1.181 | **−0.154** | 0.78 | BTC OOS Δ；holdout SOL 4/5、BNB 0/5；延遲一根四 OOS 格 2/4 |
| 6 | context_adaptive_entry_exit_cooldown | 10/10 | 1.154 | **−0.073** | 1.00 | BTC OOS −0.064、OOS2 −0.073；資金費率翻掉 BTC OOS/OOS2（基準過） |
| 7 | graded_exposure_ema_ladder_mean | 10/10 | 1.144 | **−0.070** | 1.00 | BTC OOS −0.070、OOS2 −0.066；費 0.10%+滑 0.20% 新增翻掉 BTC OOS/OOS2；B3 帶寬事前標準 2/4 |
| 8 | graded_exposure_dist_ratchet | 10/10 | 1.123 | **−0.065** | 0.90 | BTC OOS −0.065；holdout SOL 4/5、BNB 1/5（平均 −0.072）；延遲一根四 OOS 格 1/4 |
| 9 | context_adaptive_entry_whipsaw_count | 9/10 | 1.039 | **−0.176** | 0.91 | ETH IS2 掛；平均 < 冠軍；BTC 五格 Δ 全 −0.13～−0.18 |
| 10 | btc_leader_gate_relstrength_ratio | 7/10 | 0.918 | −0.042 | **0.25** | ETH IS/FULL/IS2 掛（t −2.6 / −2.3 / −3.4）；平均 < 冠軍；holdout SOL 3/5、BNB 0/5 |

---

## 3. 對冠軍逐格 ΔSharpe 與配對 t（變體 − ema50_plain）

### 3.1 BTC

| 變體 | IS | OOS | FULL | IS2 | OOS2 |
|------|----|-----|------|-----|------|
| volume_confirmed_entry_q1_ema_exit | **+0.30 (2.12)** | −0.01 (−0.34) | +0.16 (1.66) | +0.18 (1.32) | +0.02 (0.03) |
| btc_leader_gate_ema50_and | 0 | 0 | 0 | 0 | 0 |
| volume_confirmed_entry_q1_band_exit | +0.13 (0.95) | +0.04 (0.41) | +0.09 (1.03) | +0.05 (0.32) | +0.12 (1.25) |
| slow_vol_scaled_band_atr50 | +0.38 (1.81) | **−0.06 (−0.74)** | +0.17 (0.88) | +0.27 (1.23) | −0.00 (−0.31) |
| slow_vol_scaled_band_median365 | +0.36 (1.61) | **−0.15 (−1.26)** | +0.12 (0.45) | +0.18 (0.63) | −0.04 (−0.45) |
| context_adaptive_entry_exit_cooldown | **+0.27 (2.31)** | **−0.06 (−0.90)** | +0.11 (1.24) | +0.19 (1.76) | **−0.07 (−0.73)** |
| graded_exposure_ema_ladder_mean | +0.17 (1.80) | **−0.07 (−1.29)** | +0.06 (0.61) | +0.10 (0.95) | **−0.07 (−1.00)** |
| graded_exposure_dist_ratchet | +0.07 (0.51) | **−0.07 (−1.33)** | +0.00 (−0.29) | +0.02 (−0.03) | −0.05 (−0.86) |
| context_adaptive_entry_whipsaw_count | −0.13 (−1.05) | **−0.13 (−1.51)** | −0.13 (−1.72) | −0.13 (−1.11) | **−0.18 (−1.48)** |
| btc_leader_gate_relstrength_ratio | 0 | 0 | 0 | 0 | 0 |

### 3.2 ETH

| 變體 | IS | OOS | FULL | IS2 | OOS2 |
|------|----|-----|------|-----|------|
| volume_confirmed_entry_q1_ema_exit | +0.12 (1.05) | −0.02 (−0.36) | +0.06 (0.63) | +0.10 (0.83) | −0.00 (−0.08) |
| btc_leader_gate_ema50_and | +0.31 (0.40) | +0.01 (−0.13) | +0.17 (0.32) | +0.34 (0.61) | −0.02 (−0.23) |
| volume_confirmed_entry_q1_band_exit | +0.14 (1.18) | −0.01 (−0.13) | +0.07 (0.91) | +0.13 (1.12) | −0.03 (−0.24) |
| slow_vol_scaled_band_atr50 | +0.17 (0.70) | +0.02 (−0.04) | +0.10 (0.53) | +0.08 (0.25) | +0.14 (0.57) |
| slow_vol_scaled_band_median365 | +0.15 (0.55) | −0.01 (−0.23) | +0.08 (0.29) | +0.08 (0.17) | +0.07 (0.27) |
| context_adaptive_entry_exit_cooldown | +0.01 (−0.13) | +0.04 (0.24) | +0.02 (0.04) | +0.03 (−0.06) | +0.03 (0.19) |
| graded_exposure_ema_ladder_mean | +0.02 (−0.06) | +0.09 (0.92) | +0.05 (0.53) | +0.02 (0.01) | +0.16 (1.28) |
| graded_exposure_dist_ratchet | +0.08 (0.70) | +0.03 (0.29) | +0.06 (0.73) | +0.05 (0.32) | +0.09 (0.75) |
| context_adaptive_entry_whipsaw_count | −0.04 (−0.26) | +0.09 (0.97) | +0.02 (0.33) | −0.04 (−0.28) | +0.11 (0.93) |
| btc_leader_gate_relstrength_ratio | **−0.62 (−2.61)** | −0.04 (−0.64) | **−0.38 (−2.29)** | **−0.92 (−3.42)** | +0.59 (1.03) |

粗體 = 四個 OOS 格中低於 −0.05 者，或 |t| ≥ 2 者。

### 3.3 讀法

1. **本輪首次出現 |t| ≥ 2 的正向格**：BTC IS 的 `context_adaptive_entry_exit_cooldown`（+0.27，t 2.31）與 `volume_confirmed_entry_q1_ema_exit`（+0.30，t 2.12）。兩者機制相同：跳過 2020–2022 熊市裡「站上 EMA50 後 ≤ 5 根就跌回」的短命回合（冷卻版 S 類 19 回合 0 勝 Σlog −0.62；量能版跳過型 25 回合 0 勝、平均 −2.8%）。overfit 審查把 IS 的差異日按連續段拆開：量能版 BTC IS 前 3 段（8 天）佔 63%，扣掉後 t 1.14；ETH IS 前 3 段佔 98%，扣掉後 t 0.04。
2. **四個 OOS 格對冠軍沒有任何變體同時為正且 |t| ≥ 1**。最接近的是 `volume_confirmed_entry_q1_band_exit` BTC OOS2 +0.12（t 1.25），但 BTC OOS 的 +0.04 由 3 天扛（2024-04-24 / 04-14、2026-07-27），去掉前 3 個差異日即 −0.03。
3. **IS 賺 / OOS 賠的分裂跨五個視角一致**：慢帶（BTC IS +0.36～+0.38 → OOS −0.06～−0.15）、冷卻（+0.27 → −0.06/−0.07）、階梯（+0.17 → −0.07/−0.07）、棘輪（+0.07 → −0.07/−0.05）、量能 EMA 出場版（+0.30 → −0.01/+0.02）。原因都是同一個：2020–2022 的 BTC 是「熊市反彈短命」結構，任何讓進場變慢或變小的規則都賺鋸齒成本；2023 後是「假出場 → 1–3 根內再突破」與 V 型反轉結構，同一規則付動能日（冷卻版 OOS 的 D 類：2023-10-15 +43% 回合延後 1 根付 4.8%、2024-10-11 +39% 延後 3 根付 5.5%）。
4. **統計顯著的 OOS 結果三輪來仍只有負向**：本輪階梯消融 B2（門檻 0.34 兩級遲滯）BTC OOS t −2.98、OOS2 −2.54；規則 B 相對強度閘 ETH IS/FULL/IS2 t −2.6/−2.3/−3.4。

---

## 4. 敏感度（±25%，各格通過率 %）

| 變體 | n_eff BTC/ETH | BTC IS/OOS/FULL/IS2/OOS2 | ETH IS/OOS/FULL/IS2/OOS2 | 註 |
|------|--------------|--------------------------|--------------------------|----|
| volume_confirmed_entry_q1_ema_exit | 6 / 6 | 100/100/100/100/100 | 100/100/100/100/100 | 60 格全過；但 overfit 補跑 ±50%：ETH OOS2 50%（vol_win 30、q 0.5 掛）；q 掃描 BTC IS 單調升、OOS 單調降，q ∈ {0.9, 1.0} × W ∈ [10, 25] 是唯一「IS 有增益且 OOS 不劣」的小平台；ETH q=1.1/1.2 OOS+OOS2 同時掛 |
| btc_leader_gate_ema50_and | 2 / 4 | 100/100/100/100/100 | 100/**50**/100/100/**50** | ETH ema_len 38/62 十格過且 OOS 更好；btc_ema_len 38/62 ETH OOS 0.66/0.59、OOS2 0.35/0.28 四格全掛；19 點掃描 OOS 過閘帶只有 25–60，帶內 Δ ≈ 0；IS Δ 隨長度升到 75 的 +0.73 而 75 正是 OOS 最差點 |
| volume_confirmed_entry_q1_band_exit | 10 / 8 | 100/100/100/100/100 | 100/100/100/87.5/100 | ETH IS2 唯一失敗 q 0.75；atr_len 在 ETH 去重（死參數）、BTC vol_win 25 只改 2019-11 一根（去重缺口）；overfit 補跑：ETH 在 q / vol_win / k_out 三個方向的事前值都是局部谷底、相鄰值鋸齒 ±0.1 |
| slow_vol_scaled_band_atr50 | 8 / 8 | 100/75/100/100/75 | 100/87.5/100/100/100 | 失敗全在 ema_len 38/62；atr_len 38/62、k_in、k_out 十格全過 = atr_len 對通道無鑑別力 |
| slow_vol_scaled_band_median365 | 10 / 9 | 100/90/100/100/100 | 100/77.8/100/100/88.9 | 失敗幾乎全是 ema_len 38；k_in/k_out ±25% OOS+OOS2 15/16；ref_len 274/456 無鑑別力 |
| context_adaptive_entry_exit_cooldown | 7 / 7 | 100/100/100/100/100 | 100/100/100/100/100 | cooldown 0/3/5/10/20/∞：BTC IS 單調升、∞（= 通道進 + EMA50 出）最好 → IS 收益是通道效應；BTC OOS 3 最好、ETH 10–20 最好，5 兩標的都不是平台 |
| graded_exposure_ema_ladder_mean | 4 / 4 | 100/100/100/100/100 | 100/100/100/100/100 | rebalance_threshold 去重（步幅 ⅓ > 門檻）；B3 帶寬：BTC OOS/OOS2 壓到 plain 的 0.31×/0.15×，ETH 反放大 2.46×/2.06×（plain 在 ETH 的 ema_len 曲面是 V 形） |
| graded_exposure_dist_ratchet | 10 / 10 | 100/90/100/100/100 | 100/100/100/100/100 | BTC OOS 失敗 ema_len 62；s_min 0.375/0.5/0.625/1.0 IS 單調（越低越好）、OOS 平坦 |
| context_adaptive_entry_whipsaw_count | 11 / 11 | 100/90.9/100/100/90.9 | 81.8/100/100/**27.3**/100 | whip_window 45、trend_hold_min 15/25 在 BTC 映射到相同 regime（鞭打參數半死碼）；ETH IS2 基準值本身就掛 |
| btc_leader_gate_relstrength_ratio | 2 / 4 | 100/100/100/100/100 | **0/25/0/0**/100 | ratio_ema_len 在 BTC 被 is_btc 去重；ETH 四列 IS/FULL/IS2 全掛 |

三輪不變：**ema_len 是整個家族唯一的脆弱參數**，本輪 OOS 敏感度失敗仍幾乎全來自 ema_len 38/62。新增的疊加層參數三種命運：無鑑別力（atr_len、ref_len、rebalance_threshold）、單調（cooldown 在 IS、q 在 BTC）、尖點（btc_ema_len=50 在 ETH OOS，鄰值 45/55 過但 52 即掛）。

---

## 5. 壓力情境（通過格數 BTC / ETH，各 5 格）

| 變體 | 費0.10%+滑0.05% | 費0.10%+滑0.20% | 資金費率 0.01%/8h | 延遲 1 根 | 歷史資金費率 | 讀法 |
|------|----------------|----------------|------------------|----------|-------------|------|
| **ema50_plain（基準）** | 5 / 2 | 5 / 1 | 5 / 0 | 5 / 5 | 4 / 0 | BTC 前四情境 20 格全過；ETH 對持有邊距 0.02–0.05；歷史資金費率 BTC IS 0.68 掛、ETH 全掛 → 現貨執行 |
| volume_confirmed_entry_q1_ema_exit | 5 / 3 | 5 / 3 | 5 / 3 | 5 / 5 | **5 / 0** | BTC 25/25 全過（歷史資金費率 IS 0.98 ✓、基準 0.68 ✗）；ETH 成本情境掛 OOS/OOS2 與基準同命；executable 補跑：改用 Binance 現貨成交量 BTC OOS Δ 由 −0.01 變 −0.10 |
| btc_leader_gate_ema50_and | 5 / 4 | 5 / 3 | 5 / 3 | 5 / 5 | 4 / 3 | BTC 與基準逐格相同；ETH 多撐的都是 IS/FULL/IS2；延遲一根後對基準 OOS Δ 轉 −0.07/−0.04（閘門的出場提前價值集中在 BTC 跌破當天） |
| volume_confirmed_entry_q1_band_exit | 5 / 4 | 5 / 3 | 5 / 3 | 5 / **4** | **5 / 0** | 延遲一根 BTC OOS 1.31 ✓（通道版此格掉閘）但 ETH OOS 0.653 ✗ 是基準（0.82 ✓）沒有的新增失敗；現貨量重算 ETH IS2 掉閘 |
| slow_vol_scaled_band_atr50 | 5 / 5 | 5 / 5 | 4 / 4 | 5 / 5 | 4 / 1 | 換手 1/2.5 的遺傳優勢；資金費率翻 BTC OOS（1.16 vs 1.18）、ETH OOS |
| slow_vol_scaled_band_median365 | 5 / 5 | 4 / 4 | 4 / 3 | **4 / 4** | 4 / 0 | BTC OOS 對持有邊距只 0.064：滑點 0.20、資金費率、延遲一根、歷史費率四情境都翻掛（基準 20/20）；延遲一根四 OOS 格 2/4 |
| context_adaptive_entry_exit_cooldown | 5 / 5 | 5 / 2 | **3 / 0** | 5 / 5 | 3 / 0 | 資金費率翻 BTC OOS（1.15）/ OOS2（0.73）是新增失敗（邊距 0.15 vs 基準 0.22） |
| graded_exposure_ema_ladder_mean | 5 / 5 | **3 / 3** | **3 / 4** | 5 / 5 | 3 / 2 | 成交 2.1× 基準：費+滑 0.20 新增翻掉 BTC OOS（1.16）/ OOS2（0.72）；ETH 在成本情境反而多撐 |
| graded_exposure_dist_ratchet | 5 / 5 | 5 / 4 | **3 / 2** | **4 / 3** | 4 / 0 | 資金費率翻 BTC OOS/OOS2；延遲一根翻 BTC OOS、ETH OOS/OOS2 → 四 OOS 格 1/4，不是上線候選；消融 A3（EMA50 出場）延遲一根十格全過 |
| context_adaptive_entry_whipsaw_count | 4 / 4 | 2 / 2 | 2 / 1 | 3 / 2 | 0 / 2 | BTC OOS 對持有邊距 0.09，任何摩擦都翻；延遲一根四 OOS 格全掉 |
| btc_leader_gate_relstrength_ratio | 5 / 1 | 5 / 1 | 5 / 1 | 5 / 1 | 4 / 1 | ETH OOS 邊距 0.007 四情境全翻；延遲一根 0.67 → 0.51（基準反升到 0.82）= 時點敏感 |

結構性讀法：(1) 成本情境下比基準**新增**失敗格的是三個「動作變多」的變體（階梯 2.1× 成交、棘輪的資金費率/延遲、冷卻的資金費率）與慢帶 A（BTC OOS 邊距薄）；(2) 兩個量能版與 btc_leader A 在 BTC 歷史資金費率下 IS 由掛翻過（回合少 30%），是本輪唯一「壓力優於基準」的地方，但仍是 IS；(3) 延遲一根讓 ETH 五格全升（cooldown、ladder、基準本身 0.71 → 0.82）是基準遺傳的時點效應，第 2 輪已判定不可當穩健性證據。

---

## 6. 換手與曝險（FULL）

| 變體 | BTC 回合 | BTC 在場 | BTC 成交 | ETH 回合 | ETH 在場 | ETH 成交 | 註 |
|------|---------|---------|---------|---------|---------|---------|----|
| ema50_plain | 79 | 54% | 163 | 83 | 54% | 167 | 趨勢內曝險 0.94 |
| vol_atr_channel（對照） | 28 | 48% | — | 34 | 50% | — | |
| volume_confirmed_entry_q1_ema_exit | 55 | 51% | 113 | 61 | 52% | 123 | 進場腿為基準子集（變體在場而基準不在 0 根） |
| btc_leader_gate_ema50_and | 79 | 54% | 163 | 84 | 47% | 169 | ETH 與基準不同 182 根，全是「基準持有、變體空手」；回合不減，持有段被切短 |
| volume_confirmed_entry_q1_band_exit | 49 | 54% | 101 | 49 | 54% | 99 | 對通道多出 21/15 個回合 0 勝 |
| slow_vol_scaled_band_atr50 | 27 | 47% | 57 | 32 | 50% | 65 | 與通道 in_regime 只差 19/2523、21/2443 根 |
| slow_vol_scaled_band_median365 | 27 | 48% | 57 | 31 | 49% | 63 | |
| context_adaptive_entry_exit_cooldown | 60 | 51% | 123 | 65 | 52% | 131 | 冷卻根佔 12–13%；擋掉 59/67 個 plain 訊號根 |
| graded_exposure_ema_ladder_mean | 82 | 61% | **348** | 82 | 62% | **362** | 平均曝險 0.54；部分曝險日 321/304（在場日 21%）；名目換手 1.05×/0.98× 基準 |
| graded_exposure_dist_ratchet | 65 | 56% | 175 | 63 | 56% | 167 | 平均曝險 0.53；名目換手 0.67×/0.61× 基準；BTC 56%、ETH 45% 的回合從未升到滿倉 |
| context_adaptive_entry_whipsaw_count | 60 | 54% | 125 | 57 | 56% | 115 | strict 情境只佔 10–25%，61 次進場 52 次是 plain 模式 |
| btc_leader_gate_relstrength_ratio | 79 | 54% | 163 | 84 | **34%** | 169 | 閘門砍的是整段（OOS 最長 rs_off 122 根），不是鋸齒 |

---

## 7. Holdout（SOL / BNB，對 ema50_plain）

| 變體 | SOL 閘門 | SOL 平均 Δ | BNB 閘門 | BNB 平均 Δ | 判定 | 註 |
|------|---------|-----------|---------|-----------|------|----|
| ema50_plain | 5/5 | — | 3/5 | — | — | BNB IS/FULL 掛是持有 Sharpe 1.53/1.19 太高 |
| vol_atr_channel | 3/5 | +0.13 | 0/5 | −0.13 | ✗ | 第 1 輪事後補跑 |
| volume_confirmed_entry_q1_ema_exit | 5/5 | +0.022 | 3/5 | −0.011 | **pass** | 十格 \|t\| ≤ 0.94；改善在回撤（SOL 淺 1.6–4.4 pp、BNB 淺 0.5–10 pp）與回合數，不在 Sharpe；BNB FULL D2 mean R_t +0.0012 > 0（若 BNB 是主標的，事前關閉判準會觸發） |
| btc_leader_gate_ema50_and | 5/5 | **+0.307** | 4/5 | **+0.105** | **pass** | 三輪來第一個同時改善 SOL 與 BNB 的變體、BNB FULL 由掛翻過；但 SOL IS +0.50 的一半是 2021-05-18/20/21（與 ETH IS 同一事件）；還最差 3 天：SOL 五格仍 ≥ 0，BNB 五格全轉負；十格 \|t\| < 1 |
| volume_confirmed_entry_q1_band_exit | 3/5 | +0.032 | 2/5 | **−0.096** | 形式 pass（5/10、−0.033）/ 分標的 ✗ | 冠軍過閘翻掛 3 格（SOL OOS/OOS2、BNB IS2）；SOL 與通道同型；D1「量能對出場帶為正貢獻」在 SOL OOS/OOS2、BNB IS/FULL/IS2 不成立 |
| slow_vol_scaled_band_atr50 | 3/5 | +0.027 | 0/5 | −0.152 | ✗ | 3/10、平均 −0.063；輪廓與通道、A 版相同 |
| slow_vol_scaled_band_median365 | 4/5 | +0.027 | 0/5 | −0.119 | ✗ | BNB 五格 Δ 全負 |
| graded_exposure_dist_ratchet（A0） | 4/5 | +0.077 | 1/5 | −0.072 | ✗ | plain 過的 BNB IS2/OOS2 翻掛；消融 A3（EMA50 出場）SOL 5/5 +0.083、BNB 3/5 +0.018；A1（clip）SOL 5/5、BNB 2/5 −0.066 |
| btc_leader_gate_relstrength_ratio | 3/5 | −0.27 | 0/5 | −0.40 | ✗ | 四標的一致「賣出仍在漲的相對弱勢幣」；BNB FULL t −2.34 |
| cooldown / ladder / whipsaw | — | — | — | — | 未跑 | 已在 BTC OOS Δ 掛，未進 holdout |

---

## 8. 驗證結論

### 8.1 正式驗證（前 3 名，四道）

| 變體 | lookahead | overfit | executable | holdout | 關鍵理由 |
|------|-----------|---------|------------|---------|---------|
| volume_confirmed_entry_q1_ema_exit | **pass**（none） | **fail**（high） | **fail**（medium） | **pass**（low） | 前視：med 用 shift(1) 不含當根、137/133 個切點截斷 0 筆不一致、未來資料擾動前綴 0 筆改變、純迴圈重算 0 筆不一致、790 個葉值重跑 0 差異；D1 q=0 與 ema50_plain 逐位相同。過擬合：四 OOS 格 \|t\| < 0.4、三格為負；BTC IS 前 3 段（8 天）佔 63%、扣後 t 1.14；ETH IS 前 3 段佔 98%、扣後 t 0.04；q 在 BTC IS 單調升 / OOS 單調降、ETH q=1.1/1.2 兩格同時掛（刀鋒）；八個 OOS 型格（含 holdout）平均 Δ −0.021、7/8 ≤ 0；D7 事前規則觸發。可執行：安慰劑 150 次隨機置換 vr → 25.3% 隨機濾網滿足全部取代條件、BTC OOS 只在第 66–74 百分位、ETH OOS 低於隨機中位數；FULL 對基準優勢 73% 來自 3 筆跳過鞭打（2022-02-15、2021-11-25、2021-09-14），2023 起逐年 R_t 淨負；換用現貨成交量翻 7/80 個穿越日決策即讓 BTC OOS Δ −0.10、現貨/永續量比 2019 0.44 → 2026 0.11。Holdout：8/10、SOL +0.019 / BNB −0.011，\|t\| ≤ 0.94，改善在回撤與回合數 |
| btc_leader_gate_ema50_and | **pass**（none） | **fail**（high） | **fail**（medium） | **pass**（low） | 前視：BTC 序列在 `_btc_view` 截到 df.index[-1]、273 個切點 0 筆不一致、BTC 快取截短 30 根正確 raise、挖 5 根洞 ETH 對應根強制 False；BTC 分支 regime_key 與基準逐字相同。過擬合：二十格無任何 \|t\| ≥ 1；ETH IS +0.31 去 1 天（2021-05-19）剩 +0.12、去 2 天 +0.02、去 3 天 −0.06；IS2 +0.34 去 5 天 +0.004；btc_ema_len 19 點掃描 OOS 過閘帶只 25–60 且帶內 Δ ≈ 0、L=52 即掉閘、IS 最佳（75）= OOS 最差；ETH 超額 Sharpe IS +0.39 → OOS +0.06。可執行：ETH 182 個 avoided 根拆四類，OOS 早出 −0.145 + 整段壓制 −0.169 vs 中段鞭打漏掉 +0.182 + 晚進漏掉 +0.137 = +0.005（t 0.02）；OOS Sharpe +0.014 只是在場率 54→47% 的變異數縮減；2025-08-19 起 3 根 ETH 獨漲 +0.170 被擋；SOL OOS avoided 去 3 天 −0.003。Holdout：9/10、SOL +0.31 / BNB +0.11、無翻掛格；BNB 還 1 天即 IS/IS2 ≤ 0；延遲一根 SOL OOS2 Δ −0.13 |
| volume_confirmed_entry_q1_band_exit | **pass**（low） | **fail**（high） | **fail**（medium） | **pass**（high） | 前視：四標的 329–355 個切點 0 筆不一致、exit 與通道逐位相同、q=0 消融數字一致；備註 BTC vol_win 25 只改 2019-11 一根卻計為有效擾動（去重缺口）。過擬合：十格只 4 格 t ≥ 1、四 OOS 只 BTC OOS2；BTC OOS +0.04 由 3 天扛、去 3 天 −0.03；ETH 事前固定點在 q / vol_win / k_out 三方向都是谷底；5 參數六格 1.168 < 4 參數母體通道 1.19；holdout SOL OOS/OOS2 t −1.19/−1.07 是本變體最接近顯著的結果、方向為負。可執行：進場日當日報酬平均 +4.1%/+5.9%（63%/74% 落在 >+3% 日），次日續漲 +0.12%、勝率 46–47% = 量能是大陽線的雜訊代理；ETH v2−ablation 逐年 2022 一年佔 95%；現貨量重算 ETH IS2 掉閘、BTC OOS 優勢歸零；延遲一根 ETH OOS 掛是基準沒有的新失敗。Holdout：5/10 踩線、BNB −0.096、3 格由冠軍過翻掛 |

### 8.2 其餘變體的自帶檢查

| 變體 | 前視 | 其他 |
|------|------|------|
| slow_vol_scaled_band_atr50 | ABGT_DEBUG 與 `regime_atr_channel(df,50,50,1.0,0.25)` 逐位相同 | 配對回合歸因（entry/exit lead 直方圖）、ratio 桶、崩盤後 60 根；單一段 2022-08-12→18 主宰 BTC 優勢；獨立 --out 未覆寫主結果 |
| slow_vol_scaled_band_median365 | 4 標的各 48–49 個切點 0 筆不一致；enter&exit 同根 0 次 | 對通道配對 / extra / skipped 回合逐段；BTC OOS 對通道 −0.115 拆成配對 −0.069、extra −0.077、躲掉 +0.031 |
| context_adaptive_entry_exit_cooldown | cooldown=0 與 ema50_plain 逐根相同；4 個切點 0 筆不一致 | 冠軍每個回合分 A/D/S 三類；cooldown 單調性 0/3/5/10/20/∞ |
| context_adaptive_entry_whipsaw_count | 48 個切點 0 筆不一致；strict 恆真 = 通道、whip_min_count→∞ = plain 進/通道出 | A/B/C 類回合歸因、strict 佔比 |
| graded_exposure_ema_ladder_mean | 33/32 個切點 0 筆不一致；on 逐根等於 scale > 0 | B1–B4 四組消融（ratio 1.05 巢狀空假設、門檻 0.34、帶寬、單腿分解）；出場側 / 進場側 / 排列日歸因 |
| graded_exposure_dist_ratchet | 6 個切點 0 筆不一致；A4 (s_min=1, k_out=0) 與基準 regime_key / exposure_key 逐位相同 | A1/A2/A3 消融含各自 holdout 與壓力；s_min / k_out 曲面 |
| btc_leader_gate_relstrength_ratio | 28/28/25/27 個切點 0 筆不一致；`_btc_view` 與規則 A 逐字相同 | A/B 並列 avoided 日歸因、事前指定風險段 2020-10→12 命中 |

---

## 9. 每個變體教了我們什麼（排名順序，對照第 2 輪路線圖假設）

1. **volume_confirmed_entry_q1_ema_exit**：第 2 輪方向 4 假設「熊市反彈與盤整鋸齒的穿越發生在低量日」**成立但只在 IS**（BTC 跳過型 25 回合 0 勝、平均 −2.8%）；「V 型突破日量能 ≥ 1.5×」**推翻**（D4：≥1.5 桶勝率較高但平均回合報酬較低，非分級訊號；ETH 的 V 型反轉 2021-03-26、2024-04-06、2023-07-07 反而被延後 5 根付 +12%）；「濾掉日平均報酬為正則關閉」的判準 FULL 沒觸發（−0.012/−0.005）但 OOS 已是 +0.002/+0.002。消融（等拒絕率距離濾網）證明 **BTC 的量能有價格距離以外的資訊**（五格全優、FULL t 2.04；固定距離下 2×2 分出 0% 與 50% 勝率組），**ETH 沒有**（|Δ| ≤ 0.025、固定距離下量能反向）。可執行審查的安慰劑是本輪最重要的方法論發現：**任何讓進場變慢的濾網都能在 2020–2022 拿到鞭打成本，25% 的隨機濾網滿足全部取代門檻**——三輪來「六格平均高於冠軍」這條門檻一直在放行 IS 結構而非 alpha。三道驗證 pass/fail/fail/pass，是三輪最接近的候選，也是本方向的終點。
2. **btc_leader_gate_ema50_and**：第 2 輪方向 2 假設「崩盤由 BTC 帶動、ETH 的 2024-04 / 2025-02 假訊號 BTC 同步或更早出場」**部分成立**（ETH 回撤六格全淺、SOL −69% → −51%、BNB −57% → −43%、三輪第一個同時改善 SOL 與 BNB），但三個 OOS 預測（ETH OOS ≥ 0.78、OOS2 ≥ 0.52、回撤 ≤ −46%）**全未達**；ETH 的優勢是 IS 三個崩盤日（2021-05-18 −0.32、2020-03-07 −0.17、2020-09-04 −0.14，去最負 3 天轉正），OOS 正負日 +0.70/−0.70 完全抵銷、勝率 52.7%，2025/2026 ETH 獨漲被擋（2025-08-21 +13.4%）。btc_ema_len=50 在 ETH OOS 是尖點（52 即掉閘、IS 最佳 75 = OOS 最差）。與規則 B 並列證明**跨標的資訊存在且是 BTC 絕對趨勢、不是相對強度**，但它是尾部事件避險，用 Sharpe 比較看不到；若要保留，應改為預先登錄事件清單的保險成本檢定（第 12 節不列為變體方向）。
3. **volume_confirmed_entry_q1_band_exit**：把通道的進場腿換成量能確認，是第 2 輪 fastreentry 的鏡像（子集進場、共用出場）：跳過型 16/16、14/14 全輸、延後型 8/8、11/12 全付成本，淨值只是兩堆比例。對通道 V 型提早只 0–2 根（假設 2–5 推翻：V 型首日本身就 > +1 ATR）、多餘回合 0/21、0/15 勝，FULL 淨 +0.01/−0.03；D1 證明 **−0.25 ATR 出場帶單獨配 plain 進場在 BTC 五格全劣（回撤 −62%/−68%），通道的優勢是「+1 ATR 進場 × 出場帶」的組合而不是出場帶本身**；D6：ETH 濾網帳面利益全來自 5 個「20 日中位數被崩盤後爆量膨脹」的日子。唯一可證偽的具體優勢「延遲一根 BTC OOS 過」成立（1.31，通道版掉閘），但同情境 ETH OOS 掛是基準沒有的新失敗。「量能比值無量綱、比 +1 ATR 可搬」推翻（SOL 3/5 與通道同型、BNB 2/5）。
4. **slow_vol_scaled_band_atr50**：六格平均最高（1.22）卻是本輪最「無操作」的變體：atr_len 14 → 50 在通道上只改 < 1% 根（BTC 19/2523、ETH 21/2443）、回合 29/33 vs 30/35，BTC 全部優勢（IS +0.11、回撤 −46% → −37%）來自單一段 2022-08-12→18（通道進了熊市反彈 −14.7%、ATR50 仍記著 6 月崩盤所以沒進），扣掉後對通道 −0.014。合讀 A/B（plain 1.40 > 通道 1.34 = B 1.34 > A 1.24）：**帶寬越慢 BTC OOS 越差或持平，通道在 2023 後的落後不是 ATR14 崩盤後膨脹造成，是 +1 ATR 門檻本身**；A 的 365 日中位數不是多餘複雜度而是有害。方向 3 依事前規則第三條關閉。
5. **slow_vol_scaled_band_median365**：事前假設「V 型反轉提早 2–5 根」**幾乎不存在**（ratio > 1.2 的配對回合 FULL 只有 BTC 1 / ETH 5，平均 lead +0.2）——IS 的 q90 量的是 close 上穿 EMA50 的日子，通道進場（+1 ATR）發生得更晚、那時 ATR14 已衰減；真正發生的是低波動桶（ratio < 0.8）晚進 2–2.6 根、躲掉的通道回合全是輸家（5/5）。BTC IS 回撤 −38.3% 是三輪所有變體最淺，但配對回合逐視窗都輸通道（ETH OOS −0.20 log）。教訓與第 2 輪一致：**「更挑剔的通道」在 IS 是 BTC 的 alpha、在 OOS 付動能日**。
6. **context_adaptive_entry_exit_cooldown**：回答第 2 輪留下的問題——冠軍「緊接出場 5 根內的再穿越」若又在 5 根內出場，是 100% 負期望（BTC 19 回合 0 勝 Σlog −0.62、ETH 18 回合 0 勝 −0.58）；但 2023 後 BTC 的大回合（2023-10-15 +43%、2024-10-11 +39%、2025-01-10）都發生在「單日假出場 → 1–3 根再站回」之後，冷卻把它們當鋸齒、多付 ATR 膨脹期的門檻（BTC OOS D 類成本 0.30 log > S 類節省 0.19）。cooldown 曲面 BTC IS 單調升到 ∞（= 通道進場）→ IS 收益是通道效應不是冷卻效應；BTC 最佳 3、ETH 最佳 10–20，兩標的方向相反（第 2 輪 confirm_bars 分裂重演）。與第 9 名合讀，方向 1 的 (a)(b) 關閉；(c) 熊市情境版未測。
7. **graded_exposure_ema_ladder_mean**：直接攻擊 ema_len 脆弱性的假設「±25% OOS 帶寬縮到 ±0.05」在 BTC **成立**（帶寬壓到 plain 的 0.15–0.31×）、在 ETH **推翻**（放大 2.1–2.5×，因 plain 在 ETH 的 ema_len 曲面是 V 形、階梯把中心移動放大成 27–87 的有效長度）；B3 事前標準 2/4。分散紅利 +0.025 存在（三腿平均 1.119 → 變體 1.144）但沒有一格贏過 plain36（六格 1.153、十格全過）。歸因：BTC 全部優勢在空頭排列日（熊市反彈只押 ⅓，+0.139），多頭排列日淨負；ETH 相反。⅓ 倉試探把 BTC FULL 回撤 −56% → −48%（超出 ±3 pp 預期）。B1 巢狀空假設（48/50/52）就已偏離 0.03–0.05 → 階梯結構本身在 OOS 有 −0.04 的噪音；B2（兩級遲滯）BTC OOS t −2.98 是三輪最顯著的 OOS 負差。成交 2.1× 讓費+滑 0.20 新增翻掉 BTC OOS/OOS2。
8. **graded_exposure_dist_ratchet**：「鋸齒付一半成本」**成立**（≤5 根回合 Σlog 64%/51% of plain、56%/45% 的回合從未升到滿倉），但「BTC IS 回撤 ≤ −50%」**推翻**（−56.5%，頂部時已滿倉）、「BTC OOS Δ ≈ −0.03」**推翻**（−0.065：曝險介於 0–1 的日子 IS +0.151、OOS −0.175——剛站上 EMA50 的頭幾天在 2020–22 是反彈失敗、2023 後是 V 型反轉，半倉的正負號跨期翻轉）。兩個事前預測反向：A1（規格原文 clip、無棘輪）BTC 四 OOS 格全高於 A0 +0.09～+0.10、BTC 25/25 壓力全過，但 ETH IS2 掛、成交 2.3×、BNB 2/5；**A3（分級 + 冠軍 close < EMA50 出場、regime 逐位等於基準）數值上滿足全部取代門檻**（十格全過、六格 1.148、四 OOS Δ −0.040/−0.043/+0.029/+0.103、敏感度 100%、延遲一根四 OOS 4/4、holdout SOL 5/5 +0.083 / BNB 3/5 +0.018），是三輪唯一數值全達標且 holdout 兩標的都不劣的設定——但它是消融、未走三道驗證，且 BTC IS t 1.78 / OOS Δ 為負的型態與本輪被 overfit 否決的三個變體相同。
9. **context_adaptive_entry_whipsaw_count**：影子鞭打計數是**落後偵測器**：每段盤整的前兩次鞭打必以 plain 模式付掉、切到 strict 後 60 根衰減又切回（FULL 各翻轉 20 次），strict 只佔 10–25%（假設 35–60%），61 次進場 52 次是 plain 模式 → 退化成「plain 進 / −0.25 ATR 出」（D1 已證明在 BTC 五格劣於基準）。C 類回合十個視窗全 0 勝（BTC FULL 32 回合 −1.30 log）。被兩個母體同時支配（對 plain 五格負、對通道五格負），八個事前假設全落空。
10. **btc_leader_gate_relstrength_ratio**：相對強度閘在四個標的上一致地「賣出仍在漲的相對弱勢幣」：事前指定風險段 2020-10→12 完全命中（ETH +0.738 log、比值 −0.266，B 擋 63/84 日錯過 +0.782，A 一日不擋）；A 與 B 剔除的日子幾乎互斥（own_on 日一致率 0.50）。ETH OOS2 +0.59 與 IS2 −0.92 是同一機制（ETH/BTC 長熊時把 ETH 關在場外）符號隨區間翻轉。結論：規則 A 的資訊來自 BTC 絕對趨勢，不是橫斷面動能；B 對 ETH IS2 t −3.42 是三輪對基準最顯著的負差。

### 9.1 跨變體的結構性結論

1. **「改資訊集」與「改部位表示」都執行了，結果與「改濾波」相同**：三種新資訊（成交量、BTC 趨勢、慢波動尺度）與兩種新部位表示（距離分級、長度分散）產生的 Δ 型態一模一樣——BTC IS 正（+0.07～+0.38）、BTC OOS 負或零、ETH 全部在雜訊內。IS 的正差異來自 2020–2022 熊市反彈短命這個市場結構，任何減速 / 減倉 / 加條件的規則都拿得到；OOS 的負差異來自 2023 後「假出場後 1–3 根即突破」與 V 型反轉，同一批規則都付。
2. **取代門檻對「疊在 EMA50 上的任何進場規則」沒有鑑別力**：安慰劑 25.3% 通過率、六格平均含 FULL（含 IS）重複計算 IS。這解釋了為什麼三輪有 8 個變體六格平均高於冠軍但 OOS 沒有一個站得住。修正見第 13 節。
3. **成交量在 BTC 有獨立資訊、ETH 沒有**（消融 t 2.04 vs 距離濾網），但它是「進場日已是大陽線」的同根代理（進場日平均 +4.1%、次日續漲 +0.12%、勝率 46%），無法轉成 OOS 增量；且對資料來源（現貨 vs 永續量）脆弱到 7/80 個決策翻掉就換一格結論。
4. **跨標的資訊存在但是尾部事件避險**：BTC 領先閘的價值集中在 3 個 IS 崩盤日、holdout SOL/BNB 回撤各淺 16–18 pp，Sharpe 口徑看不到；相對強度是反向指標。
5. **分級曝險把二元進出的 IS/OOS 分裂搬到倉位維度**，成交數增加讓成本情境新增失敗。唯一乾淨的設定（A3：分級進場 + 棘輪 + 冠軍出場）不改 regime、只改倉位，數值全達標，但機制上仍是「半倉承受熊市反彈」。
6. **通道家族的最終定位確定**：+1 ATR 門檻本身是 BTC IS 的 alpha 與 OOS 的成本，帶寬尺度（ATR14/ATR50/中位數）無關；−0.25 ATR 出場帶單獨是負貢獻。`atr_channel.py` 維持「BTC 回撤較小、換手較低的替代品」，不升級。
7. 三輪 32 個變體、約 320 個對基準的格：正向 |t| ≥ 2 兩格（全 IS）、負向 |t| ≥ 2 六格（含 OOS）。`ema50_plain` 在「單標的日線曝險型」這個族裡是局部最優，繼續在同一族裡找變體的期望為負。

---

## 10. 冠軍決定

**champion = ema50_plain（維持）。連續 3 輪無新冠軍，依規則停止迴圈。** 十個變體對取代門檻逐項檢查：

| 變體 | 十格 ≥ 9 且四 OOS 全過 | 六格平均 > 1.09 | 四 OOS Δ ≥ −0.05 | OOS 敏感度 ≥ 0.67 | lookahead / overfit / holdout |
|------|----------------------|----------------|-----------------|------------------|------------------------------|
| volume_confirmed_entry_q1_ema_exit | ✓ 10/10 | ✓ 1.192 | ✓ −0.023 | ✓ 1.00 | pass / **fail** / pass（executable fail） |
| btc_leader_gate_ema50_and | ✓ 10/10 | ✓ 1.175 | ✓ −0.022 | ✗ 0.50 | pass / **fail** / pass（executable fail） |
| volume_confirmed_entry_q1_band_exit | ✓ 10/10 | ✓ 1.168 | ✓ −0.026 | ✓ 1.00 | pass / **fail** / pass（severity high；executable fail） |
| slow_vol_scaled_band_atr50 | ✓ 10/10 | ✓ 1.222 | ✗ −0.057 | ✓ 0.75 | 自檢 pass / — / ✗ 3/10 |
| slow_vol_scaled_band_median365 | ✓ 10/10 | ✓ 1.181 | ✗ −0.154 | ✓ 0.78 | 自檢 pass / — / ✗ 4/10 |
| context_adaptive_entry_exit_cooldown | ✓ 10/10 | ✓ 1.154 | ✗ −0.073 | ✓ 1.00 | 自檢 pass / — / — |
| graded_exposure_ema_ladder_mean | ✓ 10/10 | ✓ 1.144 | ✗ −0.070 | ✓ 1.00 | 自檢 pass / — / — |
| graded_exposure_dist_ratchet | ✓ 10/10 | ✓ 1.123 | ✗ −0.065 | ✓ 0.90 | 自檢 pass / — / ✗ 5/10（BNB −0.072） |
| context_adaptive_entry_whipsaw_count | ✗ 9/10（ETH IS2） | ✗ 1.039 | ✗ −0.176 | ✓ 0.91 | 自檢 pass / — / — |
| btc_leader_gate_relstrength_ratio | ✗ 7/10 | ✗ 0.918 | ✓ −0.042 | ✗ 0.25 | 自檢 pass / — / ✗ 3/10 |

三個走到驗證的變體 overfit 全 fail，理由可以合成一句：**對冠軍的六格增量全在 IS，四個 OOS 格 |t| < 0.4、去掉 2–3 個事件日即為零，且安慰劑分布顯示同樣的數值輪廓四分之一的隨機濾網也能達到**。holdout pass 只支持「不劣於基準」，不支持升級。`vol_atr_channel` 定位不變。`graded_exposure_dist_ratchet` 的消融 A3 數值全達標但非正式變體、未經三道驗證，不能依規則取代；它是唯一留給第 12 節的線索。

---

## 11. 引擎 / 風控模型改進提案（需改 `abgt/strategies/regime.py`）

第 2 輪 P1–P4 已全部實作（`exiting`、`scale`、`exposure_key`/`fill_key`、逐日資金費率）。以下是本輪暴露、且與「停止迴圈後轉向執行面」一致的缺口。

### P1. 運行閘門進引擎：`lag_stop` / `lag_action` / `lag_resume`

- 欄位：`RiskParams.lag_stop: float = 0.0`（策略權益 ÷ 同期持有權益 − 1 ≤ −lag_stop 時觸發；0 = 關閉）、`lag_window: int = 0`（0 = 自 start_idx 累計；> 0 = 以 N 根前為基期的滾動相對表現）、`lag_action: str = "hold"`（`hold` = 忽略 regime、目標曝險 = max_exposure 直到恢復；`flat` = 空手）、`lag_resume: str = "regime_reset"`（`regime_reset` = 趨勢先轉空再轉多；`bars` = 隔 `lag_resume_bars` 根；`ratio_high` = 策略/持有比值收盤創 `lag_resume_bars` 根新高）。
- 語意：引擎維護 `bench_eq[i] = initial × c[i] / c[start_idx]`（與 `benchmark.py` 同起算根），收盤段算 `ratio = eqv / bench_eq[i]`；觸發後在 `want` 決定前覆寫目標（hold → `target = rp.max_exposure`，flat → 0），成交仍走 pending / `exec_delay_bars`；`extra` 加 `lag_events: list[{trigger_bar, ratio, action, resume_bar}]`、`lag_bars`（受閘門控制的根數）。
- 預期：CLAUDE.md 第 1 節的運行閘門「最壞情況鎖在持有 − X% − 手續費」變成可回測的 RiskParams，用既有十格 + 壓力挑 X（IS 誤觸發率 ≤ 1 次/年、OOS 不劣化 Sharpe > 0.02），是紙上前向與 `live/` 的前提。

### P2. 分級曝險的棘輪與下限進引擎：`scale_mode` / `scale_floor`

- 欄位：`RiskParams.scale_mode: str = "track"`（`track` = 逐根跟隨 `scale[i]`，現行；`ratchet_up` = 趨勢內目標 scale 取進場以來 `scale` 的累計最大值，趨勢重置或出場成交後歸零）、`RiskParams.scale_floor: float = 0.0`（趨勢內目標曝險下限：`target = max_exposure × max(scale[i], scale_floor)`，NaN 仍視為 1）。
- 語意：`ratchet_up` 在收盤段用 `scale_hi = max(scale_hi, scale[i])`、`on=False` 或出場成交時 `scale_hi = 0`；`extra` 加 `scale_ups` / `scale_downs`（調整曝險成交的方向計數）與 `bars_to_full`（每回合從進場到 scale ≥ 0.99 的根數中位數）。
- 動機：本輪棘輪由變體檔自己遞推（A0），clip 版（A1）與棘輪版的差異是本輪最大的單一發現之一（BTC 四 OOS 格 +0.09～+0.10、成交 2.3×），但只能靠實作者手寫兩份 build；進引擎後「棘輪 vs clip」「floor 0.5 vs 0」成為 RiskParams 擾動，敏感度去重與消融自動涵蓋，第 12 節方向 1、2 才能用同一個變體檔測。

### P3. 引擎輸出成交與回合帳目：`fills_by_reason` / `notional_turnover` / `rounds` / `rel_log_vs_hold`

- 欄位（`extra`）：`fills_by_reason: dict[str,int]`（進場 / 趨勢結束 / 調整曝險 / ATR停損 / 回撤暫停 / 曝險歸零）、`notional_turnover: float`（Σ|Δqty × 成交價| ÷ 平均權益，另給年化）、`rounds: list[{entry_bar, exit_bar, entry_px, exit_px, log_ret, bars, max_scale, exit_reason}]`（regime 回合口徑，與 `Book.trips_df` 的平倉到平倉對齊）、`rel_log_vs_hold: np.ndarray`（逐日 log(eq[i]/eq[i−1]) − log(c[i]/c[i−1])，start_idx 起）。
- 動機：本輪至少 6 位實作者 / 驗證者各自手算名目換手（階梯 155.4、棘輪 101.8）、逐回合 log、逐年相對報酬、前 k 個事件日佔比；overfit 審查的三個標準檢查（去前 3 段後 t、逐年 Δ、超額 Sharpe）都只需要 `rel_log_vs_hold`。標準化後這些變成評估器一行，也是第 13 節新門檻的資料來源。

### P4. 雙標的組合引擎：`simulate_portfolio`

- 介面：`simulate_portfolio(dfs: dict[str, DataFrame], regimes: dict[str, in_regime], rp: RiskParams, weights: dict[str, float], start_idx: int)`：一本共用權益的 `Book`（`broker.py` 需支援多標的 entries）、每標的目標曝險 = `weights[s] × max_exposure × scale × 波動因子`、`rebalance_threshold` 逐標的判定、資金費率逐標的、回撤高水位與 `dd_pause` / P1 `lag_stop` 在組合層判定；基準 = 同權重買入持有（不再平衡與月再平衡各一）。索引對齊取交集、缺根視為該標的 regime False。
- 動機：實際上線交易的是 BTC+ETH（或 +SOL/BNB），單標的 Sharpe 看不到鋸齒的跨標的分散；ETH 對持有 0.045 的薄邊距在組合裡可能被 BTC 吸收，也可能拖垮；運行閘門（P1）與現貨紙上前向都應該在組合層定義。這是「部位表示」最後一個沒測的維度，且不需要新的 regime 規則。

---

## 12. 若例外開第 4 輪 / 停止後仍值得做的研究

依規則迴圈已停。以下是**唯一沒被本輪否定**、且有事前判準可寫的方向；每個都必須先在 IS + holdout SOL/BNB 登記參數，BTC/ETH OOS 不得回調，報表必列對基準與對通道的逐格 Δ 與 t、去前 3 個事件日後的 Δ、逐年 Δ、安慰劑分布（見第 13 節）。

避開（三輪合計已否定）：時間停滯出場、low 觸價停損 + 立即重進、收盤確認固定倍數 ATR 停損、月頻判定、均線交叉、家族多數決、Donchian 多尺度、EMA 階梯投票、絕對 / 相對波動目標、報酬衝擊崩盤閘、下行比值崩盤閘、共用出場的快速 OR 腿、對稱時間遲滯、影子鞭打計數、出場冷卻、慢波動尺度帶寬（中位數 / ATR50）、成交量進場確認（兩種出場）、BTC 領先 AND 閘（Sharpe 口徑）、相對強度閘、EMA 長度分散階梯、距離分級 + 出場帶。

1. **graded_exposure_ema_exit**：把本輪消融 A3 正式化為獨立變體（進場 close > EMA50 給 0.5、+0.5 ATR 升 0.75、+1 ATR 滿倉、棘輪、出場 close < EMA50 全出；regime 逐位等於冠軍、只改倉位）。假設：它的六格 1.148 / 四 OOS Δ −0.040/−0.043/+0.029/+0.103 / holdout SOL +0.083、BNB +0.018 在三道驗證下仍成立。判準（比現行門檻嚴，因為是「2 個事前出場中選 1」）：十格全過、四 OOS Δ ≥ −0.05、四 OOS 格去前 3 個差異段後 Δ ≥ −0.05 且至少一格 t ≥ 1、六格平均與四 OOS 平均 Δ 都高於安慰劑（scale 隨機置換 150 次）第 90 百分位、分標的 holdout（各平均 Δ ≥ −0.05、冠軍過閘格不翻掛）、費 0.10%+滑 0.20% 十格過、延遲一根四 OOS 格 ≥ 3。若 overfit 審查再判「增量全在 IS」則 graded_exposure 方向整個關閉。
2. **graded_exit_clip_ema50**：2×2 的缺格——二元滿倉進場（close > EMA50）、趨勢內曝險依 (close − EMA50)/ATR14 降級（d < 0.5 → 0.5、d ≥ 0.5 → 1.0，無棘輪、只降不升到 1 以上）、出場 close < EMA50。假設（由 A1 vs A0 觀察而來，須事前登記不可再調 0.5）：BTC 四 OOS 格 Δ ≥ 0（A1 的 +0.09 來自出場尾段半倉）、BTC 回撤 −56% → ≤ −50%；風險：ETH 拋物線回檔即降級（A1 ETH IS2 掛）、成交數升。判準：成交 ≤ 基準 1.6×、費 0.10%+滑 0.20% BTC 5/5、ETH IS2 過、四 OOS Δ ≥ −0.05、分標的 holdout；若 ETH IS2 掛或成交 > 1.6× 即關閉，不得調 d 門檻救。
3. **bear_context_entry_threshold**：第 2 輪方向 1 的 (c) 未測——熊市情境（EMA50 斜率為負：EMA50[i] < EMA50[i−20]，warmup 69；不用 EMA200 以免 ETH 視窗錯位）下進場要 close > EMA50 + 1 ATR，否則 plain；出場 close < EMA50（另跑 −0.25 ATR 版）。假設：strict 佔 BTC IS ≥ 35%、OOS ≤ 15%（設計上在牛市不動作）；IS 增益來自 2022 熊市反彈、OOS Δ 在 ±0.02 內。判準：十格全過、四 OOS Δ ≥ −0.02（比 −0.05 嚴，因為應近乎無操作）、IS 增益扣掉最大 3 段後仍 ≥ +0.05、對安慰劑（隨機挑同比例的 strict 日）BTC IS 在第 90 百分位以上、holdout 分標的；若 OOS strict 佔比 > 25% 或 IS 增益來自 < 3 段即關閉。這是「情境切換」最後一個假設，之後方向 1 整個關閉。
4. **running_gate_lag_stop**（依賴 P1）：不是新策略，是為 `ema50_plain` 設計運行閘門。用十格 + holdout 算冠軍對持有的滾動相對表現（累計與 90/180/365 根視窗）分布，事前登記 X 與 resume 規則，判準：IS 誤觸發 ≤ 1 次/年、觸發後十格 Sharpe ≥ plain − 0.02、最壞相對持有落後被鎖在 −(X + 成本) 內、`hold` 與 `flat` 兩種動作並列、對通道與 holdout 同樣成立。這是 CLAUDE.md 路線圖既有項目，也是上線前必做。
5. **portfolio_btc_eth_equal_weight**（依賴 P4）：BTC+ETH 各 50% 的 `ema50_plain` 組合 vs 50/50 買入持有（不再平衡與月再平衡）。假設：兩標的鋸齒相關 < 1，組合 Sharpe > 兩個單標的 Sharpe 平均、ETH 的薄邊距被吸收；判準：組合五格（IS/OOS/FULL/IS2/OOS2）Sharpe / Calmar / 回撤閘門全過、對 50/50 持有 Δ ≥ 單標的平均 Δ、成本情境（費 0.10%+滑 0.20%、歷史資金費率）組合層通過格數 ≥ 單標的；同時報告加入 SOL/BNB 各 25% 的版本作為 holdout。

---

## 13. 對評估器 / 迴圈流程的建議（不改 regime.py）

1. **取代門檻改成 OOS 口徑**：把「六格平均 Sharpe > 冠軍」改成「四個 OOS 格對冠軍的超額 Sharpe（策略 − 持有）Δ 平均 ≥ 0 且至少一格 t ≥ 1」，FULL 不計（含 IS 重複計算）。本輪三個數值全達標的變體在此門檻下全部不過，與驗證結論一致。
2. **安慰劑分布為標準診斷**：對任何「在 EMA50 上疊條件」的變體，把條件序列隨機置換 ≥ 100 次，印出隨機版滿足全部取代條件的比例與實際變體在各格的百分位；要求六格平均與四 OOS Δ 都 ≥ 第 90 百分位。本輪 25.3% 的隨機通過率說明現行門檻放行 IS 結構。
3. **事件集中度為必印欄**：對基準逐格 Δ 附「去前 1/2/3 個連續差異段後的 Δ 與 t」與逐年 Δ（P3 的 `rel_log_vs_hold` 一行可算）。本輪 BTC IS 63%、ETH IS 98%、ETH IS 閘門 2021-05-19 一天 63% 都是這一欄擋下的。
4. **資料來源壓力列**：用到價格以外序列（成交量、跨標的、資金費率）的變體，用第二來源（現貨 vs 永續、第二交易所）重算並逐格列 Δ；判準比照延遲一根（四 OOS 格 ≥ 3 過且不得新增掉閘）。本輪現貨量讓 ETH IS2 掉閘、BTC OOS Δ −0.10。
5. **holdout 分標的規則正式化**（第 2 輪 §12.2）：各標的平均 Δ ≥ −0.05、冠軍過閘格不得翻掛、四個 holdout OOS 格 Δ ≥ −0.05 同 BTC/ETH 規則；本輪 5/10、−0.033 的踩線 pass 不應再出現。
6. **壓力基準列從 `results/variants/ema50_plain.json` 逐格讀**，不用回合報告的計數（延遲一根 ETH 5/4 vs 存檔 5/5 的不一致讓 band_exit 版的 ETH OOS 新增失敗被遮掉）；任何情境下由基準過翻掛的格直接記為 executable fail。
7. **敏感度去重的 `exposure_key` 改成只雜湊落在評估視窗內的曝險**（BTC vol_win 25 只改 2019-11 一根卻計為有效擾動）；報表對每列印「regime 不同根數」。
8. **`build(df, p, ctx)` 介面**：評估器傳 `ctx = {"symbol", "aux": {...}}`，跨標的變體不必用 close 相等比例猜標的身分（lookahead 審查者指出的脆弱點）；aux 序列由評估器截到 `df.index[-1]`。
9. **ema_len 對齊的 Δ**：對變體與基準用同一 ema_len 逐點比較（本輪 band_exit 版 BTC OOS Δ 隨 ema_len 正負交錯），避免家族的 ema_len 脆弱性被讀成變體平台。
10. **消融不得事後升格**：A3 型的「事前指定對照」若要成為候選，須另立 id、走三道驗證並在 overfit 審查計入選擇效應；本輪已照此處理。
11. **多代理共用 scratchpad 的腳本用帶變體 id 的檔名**（一位驗證者的歸因腳本被同時段的另一位覆寫）。

---

## 14. 本輪產出檔案

- 變體：`experiments/variants/{slow_vol_scaled_band_atr50, slow_vol_scaled_band_median365, volume_confirmed_entry_q1_ema_exit, volume_confirmed_entry_q1_band_exit, btc_leader_gate_ema50_and, btc_leader_gate_relstrength_ratio, context_adaptive_entry_exit_cooldown, context_adaptive_entry_whipsaw_count, graded_exposure_ema_ladder_mean, graded_exposure_dist_ratchet}.py`
- 結果：`results/variants/<id>.json` 與 `.txt`；對通道：`results/variants/vs_channel/{slow_vol_scaled_band_median365, graded_exposure_dist_ratchet}.json`；診斷：`results/variants/btc_leader_gate_relstrength_ratio_diag.json`
- holdout：`results/holdout/{volume_confirmed_entry_q1_ema_exit, volume_confirmed_entry_q1_band_exit, btc_leader_gate_ema50_and, btc_leader_gate_relstrength_ratio, slow_vol_scaled_band_median365, graded_exposure_dist_ratchet}.{json,txt}`、`results/holdout/slow_vol_scaled_band_atr50_holdout/`、`results/holdout/slow_vol_scaled_band_atr50_vs_channel/`
- 驗證者腳本與消融（scratchpad，非專案檔）：verify_lookahead.py、plateau_scan.py、concentration.py、exec_check_btc_leader.py、sweep_btc_leader.py、diag_vol_band.py、trunc_test.py、scan_platform.py、attribution.py、ema_matched.py、holdout_attrib.py、ablate_graded.py、diag_ladder.py、smoke_slowband.py
- 本報告：`experiments/rounds/round_3.md`
- 迴圈狀態：`experiments/rounds/loop_state.json`（由編排端更新：champion 維持 ema50_plain、rounds_without_champion = 3 → 觸發停止規則、ledger 加 10 筆、CLAUDE.md 第 5 節加第 3 輪摘要並把路線圖「第 3 輪」勾選、「運行閘門」與「Pine 版 / 紙上前向」列為下一步）
