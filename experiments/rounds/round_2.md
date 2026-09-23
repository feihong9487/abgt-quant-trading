# ABGT 優化迴圈 — 第 2 輪回合報告

日期：2026-09-23　｜　基準：`ema50_plain`（六格平均 Sharpe 1.09、十格全過、holdout SOL 5/5 / BNB 3/5）　｜　本輪變體：10 個（5 個視角 × 2）　｜　實作失敗：0　｜　驗證：前 2 名各做 lookahead / overfit / executable / holdout 四道

## 0. 結論一句話

**無新冠軍，基準維持 `ema50_plain`。** 十個變體有四個十格全過、兩個六格平均 Sharpe 高於基準（`fastreentry_channel_or_20d_high` 1.155、`fastreentry_channel_or_3bar_confirm` 1.134），但沒有任何一個通過「四個 OOS 格對基準 ΔSharpe ≥ −0.05」：兩個快速重進版 BTC OOS 都是 −0.09～−0.10，且 overfit / executable 審查都判 fail（快速腿與通道共用出場，是對稱賭注：提早進場段由構造必勝、通道不會進的多餘回合 BTC 0/12、ETH 0/6 勝）。第 1 輪路線圖的四個方向——崩盤閘（含絕對下限）、收盤確認停損 + 重進門檻、快速重進、相對下行波動預算——本輪全部檢驗完畢、全部否定。共同原因：**日線 EMA50 本身已經是一個約一週尺度的崩盤偵測器**，疊在它上面的保護型規則不是死碼（崩盤閘 2500 根裡只改 3–5 根）就是在回檔裡低賣高買（停損），放鬆型規則（快速腿）則把通道唯一的實質優勢（回撤、換手）還回去，時間域濾波（多數決、連續根遲滯）在出場側吃掉整根崩盤日、在進場側吃掉 V 型市場的動能日。

連續兩輪無新冠軍。依 CLAUDE.md 規則「連續 3 輪無新冠軍即停」，第 3 輪是最後一輪；建議第 3 輪**改變資訊集（跨標的、成交量）或部位表示（分級曝險）**，不要再改 EMA50 的濾波邏輯——兩輪 22 個變體在 BTC/ETH 的任何 OOS 格上對基準都沒有 |t| ≥ 2 的正差異。

---

## 1. 評估設定

- 標的：BTC/USDT:USDT、ETH/USDT:USDT 日線（Binance USDT-M 永續）；holdout SOL/USDT:USDT（2020-09 起）、BNB/USDT:USDT（2020-02 起）
- 十格：IS 2020-01→2022-12、OOS 2023-01→2026-09-22、FULL；替換切分 IS2 2020-07→2023-12、OOS2 2024-01→今；× BTC/ETH
- 引擎：`regime.py` v2，只做多、收盤判定、下一根開盤成交、0.05%/邊、0.02% 滑點
- 閘門：Sharpe > 同期持有 且 Calmar > 持有 且 最大回撤 < 持有，每格各自判定
- 敏感度：每個數值參數 ±25%（映射到相同 in_regime 的擾動去重）；壓力：費 0.10%+滑 0.05%、費 0.10%+滑 0.20%、資金費率 0.01%/8h、延遲 1 根
- 同期持有 Sharpe：BTC IS 0.76 / OOS 1.18 / FULL 0.92 / IS2 1.00 / OOS2 0.75；ETH IS 1.18 / OOS 0.665 / FULL 0.93 / IS2 1.20 / OOS2 0.42
- 取代冠軍門檻：十格 ≥ 9 且四個 OOS 格全過、六格平均 Sharpe > 1.09、四個 OOS 格對冠軍 ΔSharpe ≥ −0.05、OOS 敏感度通過率 ≥ 0.67、lookahead / overfit / holdout 三道驗證 pass

本輪五個視角與對應的第 1 輪路線圖：

| 視角 | 變體 | 對應第 1 輪方向 / 提案 |
|------|------|----------------------|
| fastreentry | `fastreentry_channel_or_20d_high`、`fastreentry_channel_or_3bar_confirm` | 方向 3：冠軍 + 快速重進 |
| relvol | `relvol_ema50_downside`、`relvol_ema50_total` | 方向 4 + P3：相對下行波動預算 |
| closestop | `closestop_ema50_c3atr_hold7`、`closestop_channel_c3atr_hold7` | 方向 2 + P1：收盤確認停損 + 重進門檻 |
| crashgate | `crashgate_ret5_atr_abs`、`crashgate_dvol_ratio_abs` | 方向 1：崩盤閘 + 絕對波動下限 |
| crossasset | `crossasset_ema50_majority_vote`、`crossasset_ema50_run_hysteresis` | 跨標的可搬的無尺度規則（時間域濾波） |

---

## 2. 總表：變體 × 十格

排序：十格閘門數 → 六格平均 Sharpe。格式 Sharpe / 最大回撤，✓ = 過閘門。`vol_atr_channel`（第 1 輪最佳、已正式化為 `atr_channel.py`）列為對照。

### 2.1 BTC 五格

| # | 變體 | IS | OOS | FULL | IS2 | OOS2 |
|---|------|----|-----|------|-----|------|
| — | **ema50_plain（基準）** | 1.01 / −56% ✓ | 1.40 / −30% ✓ | 1.17 / −56% ✓ | 1.32 ✓ | 0.97 ✓ |
| 對照 | vol_atr_channel | 1.28 / −46% ✓ | 1.34 / −24% ✓ | 1.29 / −46% ✓ | 1.50 / −46% ✓ | 0.94 / −24% ✓ |
| 1 | fastreentry_channel_or_20d_high | 1.17 / −58% ✓ | 1.30 / −28% ✓ | 1.21 / −58% ✓ | 1.37 / −58% ✓ | 0.93 / −28% ✓ |
| 2 | fastreentry_channel_or_3bar_confirm | 1.10 / −53% ✓ | 1.30 / −33% ✓ | 1.18 / −53% ✓ | 1.31 / −53% ✓ | 0.91 / −33% ✓ |
| 3 | relvol_ema50_downside | 0.89 / −56% ✓ | 1.40 / −33% ✓ | 1.12 / −56% ✓ | 1.23 / −56% ✓ | 0.96 / −33% ✓ |
| 4 | relvol_ema50_total | 0.98 / −56% ✓ | 1.41 / −29% ✓ | 1.16 / −56% ✓ | 1.29 / −56% ✓ | 0.99 / −29% ✓ |
| 5 | closestop_ema50_c3atr_hold7 | 0.88 / −56% ✓ | 1.24 / −38% ✓ | 1.03 / −56% ✓ | 1.24 / −56% ✓ | 0.82 / −38% ✓ |
| 6 | crashgate_ret5_atr_abs | 1.03 / −56% ✓ | 1.34 / −36% ✓ | 1.15 / −56% ✓ | 1.33 / −56% ✓ | 0.88 / −36% ✓ |
| 7 | crashgate_dvol_ratio_abs | 0.98 / −56% ✓ | 1.37 / −30% ✓ | 1.15 / −56% ✓ | 1.29 / −56% ✓ | 0.94 / −30% ✓ |
| 8 | crossasset_ema50_majority_vote | 1.06 / −59% ✓ | 1.37 / −34% ✓ | 1.18 / −59% ✓ | 1.28 / −59% ✓ | 1.03 / −34% ✓ |
| 9 | crossasset_ema50_run_hysteresis | 1.06 / −58% ✓ | 1.21 / −35% ✓ | 1.11 / −58% ✓ | 1.26 / −58% ✓ | 0.85 / −35% ✓ |
| 10 | closestop_channel_c3atr_hold7 | 1.10 / −46% ✓ | 1.13 / −33% ✗ | 1.10 / −46% ✓ | 1.37 / −46% ✓ | 0.70 / −33% ✗ |

### 2.2 ETH 五格

| # | 變體 | IS | OOS | FULL | IS2 | OOS2 |
|---|------|----|-----|------|-----|------|
| — | **ema50_plain（基準）** | 1.26 / −61% ✓ | 0.71 / −51% ✓ | 0.99 / −61% ✓ | 1.22 ✓ | 0.45 ✓ |
| 對照 | vol_atr_channel | 1.43 / −46% ✓ | 0.71 / −48% ✓ | 1.09 / −48% ✓ | 1.32 / −46% ✓ | 0.55 / −48% ✓ |
| 1 | fastreentry_channel_or_20d_high | 1.40 / −48% ✓ | 0.77 / −50% ✓ | 1.09 / −50% ✓ | 1.38 / −49% ✓ | 0.52 / −50% ✓ |
| 2 | fastreentry_channel_or_3bar_confirm | 1.40 / −51% ✓ | 0.74 / −52% ✓ | 1.08 / −52% ✓ | 1.34 / −52% ✓ | 0.58 / −52% ✓ |
| 3 | relvol_ema50_downside | 1.36 / −61% ✓ | 0.72 / −51% ✓ | 1.05 / −61% ✓ | 1.27 / −61% ✓ | 0.46 / −51% ✓ |
| 4 | relvol_ema50_total | 1.25 / −61% ✓ | 0.72 / −51% ✓ | 0.99 / −61% ✓ | 1.21 / −61% ✓ | 0.46 / −51% ✓ |
| 5 | closestop_ema50_c3atr_hold7 | 1.57 / −55% ✓ | 0.68 / −52% ✓ | 1.15 / −55% ✓ | 1.51 / −55% ✓ | 0.42 / −52% ✗ |
| 6 | crashgate_ret5_atr_abs ¹ | 1.05 / −61% ✓ | 0.69 / −53% ✓ | 0.87 / −61% ✓ | 1.22 / −61% ✓ | 0.42 / −53% ✗ |
| 7 | crashgate_dvol_ratio_abs ¹ | 0.99 / −61% ✗ | 0.75 / −51% ✓ | 0.86 / −61% ✓ | 1.23 / −61% ✓ | 0.48 / −51% ✓ |
| 8 | crossasset_ema50_majority_vote | 1.47 / −56% ✓ | 0.58 / −64% ✗ | 1.05 / −64% ✓ | 1.41 / −56% ✓ | 0.34 / −64% ✗ |
| 9 | crossasset_ema50_run_hysteresis | 1.37 / −65% ✓ | 0.64 / −57% ✗ | 1.02 / −65% ✓ | 1.32 / −65% ✓ | 0.27 / −57% ✗ |
| 10 | closestop_channel_c3atr_hold7 | 1.66 / −47% ✓ | 0.65 / −53% ✗ | 1.19 / −53% ✓ | 1.53 / −47% ✓ | 0.48 / −53% ✓ |

¹ 兩個崩盤閘 warmup 78/79 > 基準 49，ETH 快取從 2019-11-27 起（2020-01-01 前只有 35 根），ETH IS/FULL 起點落在 2020-02-13/14（基準 2020-01-15），錯過 2020-01/02 的 +70% 段。閘門判定仍是同視窗（持有也從同一根起算）所以有效；但評估器印的 ETH IS/FULL 對基準 Δ（−0.21 / −0.27）是跨視窗數字。同視窗重算：`crashgate_ret5_atr_abs` IS Δ 0.000、FULL −0.009；`crashgate_dvol_ratio_abs` IS −0.034、FULL −0.003。

### 2.3 綜合：閘門數與取代冠軍門檻檢查

| # | 變體 | 十格 | 六格 | 六格平均 Sharpe | 四 OOS 格 Δ 最小 | OOS 敏感度最低 | 取代門檻未過的項目 |
|---|------|------|------|----------------|-----------------|---------------|-------------------|
| — | ema50_plain | 10/10 | 6/6 | 1.09 | — | 1.00 | — |
| 1 | fastreentry_channel_or_20d_high | 10/10 | 6/6 | **1.155** | −0.098 | 1.00 | OOS Δ（BTC OOS −0.10）；overfit ✗、executable ✗ |
| 2 | fastreentry_channel_or_3bar_confirm | 10/10 | 6/6 | 1.134 | −0.093 | 0.90 | OOS Δ（BTC OOS −0.09、OOS2 −0.07）；overfit ✗、executable ✗ |
| 3 | relvol_ema50_downside | 10/10 | 6/6 | 1.089 | −0.013 | 1.00 | 平均 Sharpe 未高於冠軍（1.089 vs 1.092）；BTC IS/FULL/IS2 t ≈ −2.25 |
| 4 | relvol_ema50_total | 10/10 | 6/6 | 1.086 | +0.008 | 1.00 | 平均 Sharpe 未高於冠軍；七年只 3/2 次減倉成交 = 無操作 |
| 5 | closestop_ema50_c3atr_hold7 | 9/10 | 6/6 | 1.094 | −0.157 | 0.00² | ETH OOS2 掛（差 0.004）；BTC OOS/OOS2 Δ −0.16 |
| 6 | crashgate_ret5_atr_abs | 9/10 | 6/6 | 1.022 | −0.093 | 0.75 | ETH OOS2 掛（差 0.0002）；平均 < 冠軍；BTC OOS/OOS2 Δ −0.06/−0.09 |
| 7 | crashgate_dvol_ratio_abs | 9/10 | 5/6 | 1.017 | −0.033 | 0.875 | ETH IS 掛（視窗錯位）；平均 < 冠軍；BTC 五格全負 |
| 8 | crossasset_ema50_majority_vote | 8/10 | 5/6 | 1.117 | −0.138 | 0.333 | ETH OOS / OOS2 掛；ETH OOS2 敏感度 33% |
| 9 | crossasset_ema50_run_hysteresis | 8/10 | 5/6 | 1.069 | −0.184 | 0.50 | ETH OOS / OOS2 掛；四個 OOS 格 Δ 全 < −0.05 |
| 10 | closestop_channel_c3atr_hold7 | 7/10 | 4/6 | 1.137 | −0.271 | 0.00 | BTC OOS / OOS2、ETH OOS 掛；BTC OOS 敏感度 0% |

² 評估器以 regime_key 去重後只剩 ema_len 兩組且都掛；含停損參數全 8 列 BTC OOS 62.5%（見第 4 節）。

---

## 3. 對基準逐格 ΔSharpe 與配對 t（變體 − ema50_plain）

### 3.1 BTC

| 變體 | IS | OOS | FULL | IS2 | OOS2 |
|------|----|-----|------|-----|------|
| fastreentry_channel_or_20d_high | +0.15 (0.98) | **−0.10 (−0.99)** | +0.03 (0.11) | +0.05 (0.21) | −0.05 (−0.46) |
| fastreentry_channel_or_3bar_confirm | +0.09 (0.42) | **−0.09 (−0.96)** | +0.00 (−0.24) | −0.01 (−0.20) | **−0.07 (−0.57)** |
| relvol_ema50_downside | **−0.12 (−2.24)** | −0.00 (−0.67) | **−0.06 (−2.29)** | **−0.09 (−2.24)** | −0.01 (−0.67) |
| relvol_ema50_total | −0.04 (−1.35) | +0.02 (−0.32) | −0.01 (−1.30) | −0.03 (−1.50) | +0.01 (−0.32) |
| closestop_ema50_c3atr_hold7 | −0.13 (−1.10) | **−0.16 (−1.58)** | −0.14 (−1.81) | −0.08 (−0.90) | **−0.16 (−1.16)** |
| crashgate_ret5_atr_abs | +0.02 (1.14) | **−0.06 (−1.01)** | −0.02 (−0.74) | +0.02 (1.14) | **−0.09 (−1.01)** |
| crashgate_dvol_ratio_abs | −0.03 (−0.56) | −0.02 (−1.02) | −0.03 (−0.82) | −0.03 (−0.56) | −0.03 (−1.02) |
| crossasset_ema50_majority_vote | +0.05 (0.31) | −0.03 (−0.35) | +0.00 (0.07) | −0.04 (−0.10) | +0.06 (0.25) |
| crossasset_ema50_run_hysteresis | +0.05 (0.33) | **−0.18 (−1.61)** | −0.06 (−0.57) | −0.05 (−0.26) | **−0.12 (−0.87)** |
| closestop_channel_c3atr_hold7 | +0.09 (0.03) | **−0.27 (−1.71)** | −0.08 (−1.02) | +0.05 (−0.19) | **−0.27 (−1.25)** |

### 3.2 ETH

| 變體 | IS | OOS | FULL | IS2 | OOS2 |
|------|----|-----|------|-----|------|
| fastreentry_channel_or_20d_high | +0.14 (0.66) | +0.05 (0.32) | +0.10 (0.73) | +0.16 (0.86) | +0.07 (0.40) |
| fastreentry_channel_or_3bar_confirm | +0.14 (0.98) | +0.03 (0.13) | +0.09 (0.83) | +0.12 (0.89) | +0.12 (0.75) |
| relvol_ema50_downside | +0.10 (0.37) | +0.01 (0.00) | +0.06 (0.36) | +0.05 (−0.05) | +0.01 (0.00) |
| relvol_ema50_total | −0.01 (−0.85) | +0.01 (−0.02) | −0.00 (−0.79) | −0.01 (−0.90) | +0.01 (−0.02) |
| closestop_ema50_c3atr_hold7 | +0.31 (1.36) | −0.03 (−0.40) | +0.16 (0.97) | +0.29 (1.34) | −0.03 (−0.29) |
| crashgate_ret5_atr_abs | 0.00 ¹ (−1.11) | −0.02 (−1.02) | −0.01 ¹ (−1.04) | 0.00 (−0.09) | −0.03 (−1.02) |
| crashgate_dvol_ratio_abs | −0.03 ¹ (−1.02) | +0.04 (1.87) | −0.00 ¹ (−0.19) | +0.01 (1.45) | +0.03 (1.26) |
| crossasset_ema50_majority_vote | +0.21 (1.23) | **−0.14 (−0.66)** | +0.06 (0.59) | +0.18 (1.20) | **−0.12 (−0.42)** |
| crossasset_ema50_run_hysteresis | +0.11 (0.84) | **−0.08 (−0.44)** | +0.03 (0.39) | +0.10 (0.82) | **−0.18 (−0.84)** |
| closestop_channel_c3atr_hold7 | +0.40 (1.18) | **−0.06 (−0.53)** | +0.19 (0.71) | +0.30 (0.89) | +0.03 (−0.04) |

¹ 同視窗重算值（見 2.2 註）。粗體 = 四個 OOS 格中低於 −0.05 門檻者。

讀法：100 格裡 |t| ≥ 2 的只有 `relvol_ema50_downside` 的 BTC IS / FULL / IS2（全負、t −2.24～−2.29），是本輪唯一統計顯著的結果，方向是劣化。所有正向 Δ 的 |t| 最大 1.87（`crashgate_dvol_ratio_abs` ETH OOS，且該變體與基準只差 5 根）。四個 OOS 格對基準 Δ 全部 ≥ −0.05 的只有 relvol 兩個變體，而它們是「幾乎不動作」的變體（趨勢內 2–7% 的日子砍倉）。

### 3.3 BTC / ETH 分裂（跨四個家族一致）

| 變體 | 幫 ETH 的地方 | 傷 BTC 的地方 |
|------|-------------|-------------|
| relvol_ema50_downside | ETH IS +0.10（2020-02 COVID 前砍 +9.6%、2020-09 +11.3%、2021-05 趨勢重進以 0.37 曝險躲鞭打 +12.6%） | BTC IS −0.12、t −2.24（每次砍倉都在牛市加速段：2021-01-21→02-09 砍到 0.56、價格 +53%、−11.9%） |
| closestop_ema50_c3atr_hold7 | ETH IS +0.31（2021-05-19 單日 +0.32 log = 56%） | BTC 五格全負（無任何頂部高於 EMA50 + 3 ATR 再崩；6 次冷卻重進 100% 在高於停損價 +8.6% 處） |
| closestop_channel_c3atr_hold7 | ETH IS +44%（2021-05-18 +34.4%、2020-09-04 +13.9%） | BTC 對通道期末權益比 0.75（10 次回檔停損各 −4～−12%） |
| fastreentry ×2 | ETH 快速腿回合對數合計 +2.08（2020-10-11 +279%、2025-06-30 +69%） | BTC 快速腿 23 回合勝率 26%、IS2 勝率 9%；2022 熊市 8 次全輸 |
| crashgate_dvol_ratio_abs（無價格解除消融） | ETH 五格全過、OOS +0.14、OOS2 +0.16 | BTC IS/IS2 −0.11（2021-01-27 旗標 30.4k → ~40k 重進） |

第 1 輪 `vol_downside_gate` 的「ETH 全正、BTC 全負」不是那個變體的特性，是 ETH 與 BTC 歷史結構的差異：ETH 有「拋物線頂部 → 崩盤」（2020-09、2021-05）與「緩慢 V 轉」（2020-10、2023-01），BTC 的頂部從不比 EMA50 高出 3 ATR、熊市反彈更頻繁。任何在 ETH 事件上賺錢的規則都在 BTC 結構上付錢，這不是參數問題。

---

## 4. 敏感度（±25%，各格通過率 %）

| 變體 | n_eff | BTC IS/OOS/FULL/IS2/OOS2 | ETH IS/OOS/FULL/IS2/OOS2 | 註 |
|------|-------|--------------------------|--------------------------|----|
| fastreentry_channel_or_20d_high | 9 / 10 | 100/100/100/100/100 | 100/100/100/100/100 | hi_len 15/25 十格全過；±50%：BTC OOS 78%、ETH OOS 90%（失敗在 ema_len 25/75、hi_len 30/10）；hi_len 5–40 對通道 BTC IS/FULL/IS2 全負、沒有平台 |
| fastreentry_channel_or_3bar_confirm | 10 / 10 | 100/90/100/100/100 | 100/100/100/100/100 | 唯一失敗 BTC OOS ema_len=62；confirm_bars 2/3/4：BTC 單調遞增（N≥8 即等於通道）、ETH 平坦；兩標的最佳 N 方向相反 |
| relvol_ema50_downside | 2 ² | 100/100/100/100/100 | 100/100/100/100/100 | sizing 四參數被 regime_key 去重；不去重 20 組：BTC OOS 95%（ema_len 75）、ETH IS2 95%（vol_target 0.7）；vol_target 掃描 BTC IS 谷形（0.7→1.01、1.4→0.89、2.1→0.97） |
| relvol_ema50_total | 2 ² | 100/100/100/100/100 | 50/100/100/50/100 | ETH IS/IS2 掛 ema_len=38（基準同一弱點）；sizing 8 組不去重十格 100%；vol_target 1.05 在 ETH IS 升到 1.39、BTC IS 降到 0.95 |
| closestop_ema50_c3atr_hold7 | 2 ² | 100/0/100/100/0 | 100/100/100/100/100 | 去重後只剩 ema_len 38/62 且兩組 BTC OOS/OOS2 都掛（基準同擾動 100% 過 = 停損拿掉了基準的 ema_len 穩健性）；全 8 列 BTC OOS 62.5% / OOS2 62.5%、ETH OOS 87.5% / OOS2 75%；k 曲面 BTC 單調（3.75 逼近基準）、ETH IS k=3 尖峰 |
| crashgate_ret5_atr_abs | BTC 8/12、ETH 6/12 | 100/87.5/100/100/75 | 66.7/83.3/83.3/50/83.3 | k_rel、ref_len 無鑑別力；OOS 失敗全在 ema_len；k_trig 2.25 兩標的 IS/OOS 沒有共同好值 |
| crashgate_dvol_ratio_abs | 8/14 | 100/100/100/100/87.5 | 25/100/62.5/62.5/87.5 | ratio_rel / abs_rel / ref_len 全無效（價格解除先觸發，波動解除路徑是死碼）；ETH IS 失敗全在錯位視窗 |
| crossasset_ema50_majority_vote | 6 / 6 | 100/50/100/100/100 | 100/66.7/100/100/33.3 | ETH OOS2 只有 ema_len 38/62 過，window / min_above 任一方向都掛：(5,3) 是尖點不是平台 |
| crossasset_ema50_run_hysteresis | 2 | 100/50/100/100/100 | 50/50/50/0/50 | 只 ema_len 生效（n=2 ±25% 四捨五入回 2）；ema_len=62 ETH 五格全掛 |
| closestop_channel_c3atr_hold7 | BTC 7/12、ETH 8/12 | 100/0/100/100/0 | 100/50/100/100/87.5 | 全 12 列 BTC OOS 1/12（只 atr_stop_mult 3.75 過）、OOS2 1/12；ETH OOS 6/12 |

² 評估器以 in_regime 雜湊去重，RiskParams 類參數（vol_target、atr_stop_mult、reentry_min_bars…）的擾動不改 in_regime 全被判 same_regime，四個變體的 summary 只反映 ema_len；那些變體的真正敏感度要看「註」欄的不去重數字。這是評估器缺口（第 10 節 P3、第 12 節）。

跨兩輪不變的事實：**ema_len 是整個 EMA50 家族唯一的脆弱參數**，本輪所有 OOS 敏感度失敗幾乎都來自 ema_len 38/62/75；疊加層的參數要嘛無鑑別力（k_rel、ref_len、atr_len）、要嘛單調（confirm_bars、atr_stop_mult）。

---

## 5. 壓力情境（通過格數 BTC / ETH，各 5 格）

| 變體 | 費0.10%+滑0.05% | 費0.10%+滑0.20% | 資金費率 0.01%/8h | 延遲 1 根 | 讀法 |
|------|----------------|----------------|------------------|----------|------|
| **ema50_plain（基準）** | 5 / 2 | 5 / 1 | 5 / 0 | 5 / 4 | BTC 20 格全過；ETH 對持有邊距只 0.02–0.05，任何額外摩擦都翻（資金費率下 ETH 五格全掛） |
| fastreentry_channel_or_20d_high | 5 / 5 | 5 / 5 | 4 / 3 | **3 / 3** | 資金費率翻 BTC OOS、ETH OOS/OOS2；延遲翻四個 OOS 格全部（快速腿在大陽線當天觸發，隔日成交吃掉邊際；通道 BTC OOS2 延遲後仍過、本變體不過） |
| fastreentry_channel_or_3bar_confirm | 5 / 5 | 5 / 5 | 3 / 4 | 4 / 3 | 資金費率翻 BTC OOS/OOS2、ETH OOS；延遲翻 BTC OOS、ETH OOS/OOS2 |
| relvol_ema50_downside | 5 / 4 | 5 / 2 | 5 / 2 | 5 / 4 | ETH 失敗全是基準遺傳；變體在資金費率下 ETH 反而比基準多撐兩格 |
| relvol_ema50_total | 5 / 3 | 5 / 1 | 5 / 1 | 5 / 4 | 同上；沒有任何一格由基準 pass 翻成 fail |
| closestop_ema50_c3atr_hold7 | 4 / 3 | 2 / 3 | 2 / 3 | 5 / 5 ³ | 三組成本情境把 BTC OOS/FULL/OOS2 翻掉；³ 延遲一根不可採信（引擎缺口，見 P1） |
| crashgate_ret5_atr_abs | 5 / 0 | 3 / 0 | 3 / 0 | 5 / 4 | ETH 三組成本情境十格全掛（邊距 0.02–0.03）；BTC OOS 高滑點 1.178 / 資金費率 1.147 vs 持有 1.179 |
| crashgate_dvol_ratio_abs | 5 / 3 | 5 / 0 | 5 / 0 | 5 / 4 | 與基準同命；變體在費+滑0.05 的 ETH OOS/OOS2 反而多撐兩格 |
| crossasset_ema50_majority_vote | 5 / 3 | 5 / 3 | 4 / 3 | 4 / 5 | BTC 資金費率 OOS 剛好 1.18 = 持有；延遲一根反而讓 ETH OOS 升到 0.77 過閘 = 結果被少數日子主宰 |
| crossasset_ema50_run_hysteresis | 4 / 3 | 3 / 3 | 3 / 2 | 4 / 5 | BTC OOS 四情境全掉（1.17/1.10/1.02/1.09 vs 1.18）；延遲一根 ETH 反而過（同上，時點運氣） |
| closestop_channel_c3atr_hold7 | 3 / 4 | 3 / 3 | 3 / 3 | 4 / 4 | 基準已掛三格四情境全掛；ETH OOS2 在滑點 0.20 與資金費率翻掛 |

³ `stop_trigger="close"` 且 `exec_delay_bars>0` 時，收盤觸發後那一根部位仍 > 0（出場排在 i+2），引擎照 `on=True` 排一筆新進場到 i+3，繞過 `wait_price` 與 `reentry_min_bars`；BTC 7/17、ETH 9/17 個停損在出場成交後 1 根就重進，停損計數膨脹到 27/25。修正提案見第 10 節 P1。

兩個結構性讀法：(1) 只有快速重進兩版在成本情境下比基準**新增**失敗格（BTC 資金費率 / 延遲），其餘變體的壓力失敗都是 ETH 基準邊距薄的遺傳；(2) 延遲一根讓時間域濾波（MV、RH）的 ETH OOS 由掛翻過，是「結果由特定日子決定」的訊號，不是穩健性。

---

## 6. 換手與曝險（FULL）

| 變體 | BTC 回合 | BTC 在場 | ETH 回合 | ETH 在場 | 停損 / 暫停 / 其他 |
|------|---------|---------|---------|---------|-------------------|
| ema50_plain | 79 | 54% | 83 | 54% | 0 / 0 |
| vol_atr_channel（對照） | 28 | 48% | 34 | 50% | 0 / 0 |
| fastreentry_channel_or_20d_high | 40 | 51% | 40 | 52% | 0 / 0；BTC 41 次進場 23 次只靠快速腿、ETH 16/41 |
| fastreentry_channel_or_3bar_confirm | 42 | 51% | 40 | 52% | 0 / 0；BTC 44 次進場 23 次純快速腿、ETH 18/41 |
| relvol_ema50_downside | 79 | 53% | 83 | 53% | 調倉 10 / 7 次（減 6 / 5） |
| relvol_ema50_total | 79 | 53% | 83 | 53% | 減倉成交 3 / 2 次（七年） |
| closestop_ema50_c3atr_hold7 | 80 | 51% | 80 | 52% | 停損 17 / 17（BTC 11 重置 + 6 冷卻；ETH 13 + 4） |
| crashgate_ret5_atr_abs | 81 | 54% | 84 | 54% | 崩盤日 3 / 1（趨勢內觸發 2 / 1 次） |
| crashgate_dvol_ratio_abs | 80 | 54% | 83 | 54% | 與基準 in_regime 不同 5 / 5 根（2500 根中） |
| crossasset_ema50_majority_vote | 39 | 53% | 40 | 54% | 0 / 0；狀態翻轉 81 vs 基準 163 |
| crossasset_ema50_run_hysteresis | 50 | 54% | 46 | 54% | 0 / 0；翻轉 103 / 93 vs 163 / 167 |
| closestop_channel_c3atr_hold7 | 37 | 45% | 39 | 48% | 停損 21 / 19（BTC 11 次與通道出場同根、10 次回檔停損） |

觀察：疊加在基準上的「保護型」規則（relvol、crashgate、closestop_ema50）回合數與在場率跟基準幾乎一樣——它們不是新的交易，是把 EMA50 出場提前 0–1 根或在 2500 根裡改 5 根；砍換手的只有通道家族（+1 ATR 進場）與時間域濾波，後者付的是崩盤日與動能日。

---

## 7. 驗證結論

### 7.1 正式驗證（前 2 名，四道）

| 變體 | lookahead | overfit | executable | holdout | 關鍵理由 |
|------|-----------|---------|------------|---------|---------|
| fastreentry_channel_or_20d_high | **pass**（none） | **fail**（high） | **fail**（medium） | **pass**（medium）6/10 | 純 Python 重算 EMA/ATR/highest/狀態機 0 筆不一致；190/186 個切點截斷 0 筆不一致；結果檔逐位可重現。過擬合：對冠軍十格 \|t\| < 1、對母體通道 BTC 四格 t 負；5 參數換到六格 1.155 卻比自己的 4 參數母體 1.19 低；逐段歸因：BTC 提早進場 11/11 贏 +0.41 log、多餘回合 0/12 贏 −0.57 log，淨 −0.16；hi_len 5–40 對通道 BTC 全負無平台。可執行：快速腿每次觸發期望 BTC −0.6% / ETH +0.2%、2022 熊市 BTC 7/7 輸；延遲一根四 OOS 格全掉。Holdout：SOL 5/5（Δ +0.14）、BNB 1/5（五格 Δ 全負、平均 −0.11）；平均 +0.018 形式上過，但 SOL 上 70% 進場只靠快速腿、回撤退回 plain 水準（−58% → −70%） |
| fastreentry_channel_or_3bar_confirm | **pass**（low） | **fail**（high） | **fail**（medium） | **pass**（medium）6/10 | 913/894 個切點截斷 0 筆不一致；獨立迴圈重算 0 筆不一致；結果檔逐字元可重現。過擬合：confirm_bars→∞ 的 regime_key 等於 vol_atr_channel（巢狀特例），對這個空假設 BTC 五格 Δ −0.18/−0.04/−0.11/−0.19/−0.03、t −1.50/−0.03/−1.09/−1.62/0.00；confirm_bars 在 BTC 單調（N≥8 = 通道）→ 參數應移除；快速腿扣前兩筆 BTC −0.54 log；ema_len 在 ETH OOS 鋸齒（45→0.94、50→0.74、55→0.88）。可執行：配對到通道的增量只 +4–5%/次（評估摘要把 2024-01-28 +50.9% 記在快速腿頭上，實際增量 +5.3%）；無配對鞭打 BTC 14/23、ETH 6/18；BTC 全期終值比純通道少 22.6%。Holdout：SOL 3/5、BNB 3/5，十格 Δ 平均 −0.001、\|t\| ≤ 1.09，等於「在沒碰過的標的上與純 EMA50 等價」；基準同 holdout 8/10，變體少過兩格 |

### 7.2 其餘變體的自帶檢查（實作者在評估內做的）

| 變體 | 前視 | 其他 |
|------|------|------|
| relvol_ema50_downside | 28 個切點截斷 dvol/ref/target/equity 0 筆不一致；獨立重算與引擎差 < 1e-14 | holdout SOL 5/5、BNB 3/5、十格 Δ 全 ≥ 0、平均 +0.027（pass）；只取曝險 < 1 日子的配對 t 與全樣本相同（BTC IS −2.27，非稀釋） |
| relvol_ema50_total | 43/41 個切點 0 筆不一致 | 與主變體並排：BTC 五格對照勝、ETH IS/FULL/IS2 主變體勝、OOS 持平；A/B/C 日成本歸因 |
| closestop_ema50_c3atr_hold7 | — | 停損逐次歸因（重置 / 冷卻路徑）；k 曲面；發現引擎延遲 × 收盤停損缺口 |
| closestop_channel_c3atr_hold7 | regime_key 與 vol_atr_channel 兩標的完全一致 | 停損逐次拆解（同根 / 回檔 / 提前）；holdout 預覽 SOL 4/5、BNB 1/5（scratchpad，非正式） |
| crashgate_ret5_atr_abs | 9/9 個切點 0 筆不一致 | 同視窗基準重跑（ETH IS Δ 0.000）；PRICE_RELEASE 消融；k_trig 消融 |
| crashgate_dvol_ratio_abs | — | 三段消融（純比值 / 無價格解除 / 全套）；同視窗基準重跑；warmup=49 對齊替代（十格全過但六格 1.08 < 1.09） |
| crossasset_ema50_majority_vote | 26 個切點 0 筆不一致；w=k=1 與 ema50_plain 逐根相同 | OOS 逐年、最差相對日名單 |
| crossasset_ema50_run_hysteresis | BTC 0/69、ETH 0/66 切點不一致；n=1 與 ema50_plain 逐根相同 | 晚進 / 晚出日報酬差拆解；IS run-length 分布 |

---

## 8. 每個變體教了我們什麼（排名順序，對照第 1 輪假設）

1. **fastreentry_channel_or_20d_high**：第 1 輪方向 3 假設「V 型行情少付 1 ATR、盤整市快速腿沉默」——前半成立（BTC 2023-01 +27%、2025-04 +17%、ETH 2020-10 +279%、2023-01/10），後半**推翻**：熊市反彈與盤整裡 EMA50 本身就在 20 日高附近，「站回 EMA50 且創 20 日新高」很常見，BTC 2022 年 8 次快速進場全輸、2024 盤整 4 次全輸。逐段歸因證明快速腿是與通道共用出場的對稱賭注（提早段必勝 +0.5 ATR、多餘段必敗 −(2.4% + 0.25 ATR + 費)），期望取決於 V 轉與熊市反彈的比例，沒有結構性 edge。通道的 OOS 落後不是「1 ATR 太貴」，是 1 ATR 擋掉的假進場 20 日新高擋不掉。額外教訓：在 SOL 上 70% 進場只靠快速腿，變體退化成「EMA50 + 20 日新高」的新策略，通道腿冗餘。
2. **fastreentry_channel_or_3bar_confirm**：時間確認確實抓到通道抓不到的低動能站上（進場時 (c−ema)/atr 中位 0.48），但兩標的性質相反——ETH 是「緩慢 V 轉」標的（快速腿 18 回合 +2.08 log > 通道腿）、BTC 是「盤整後突破」標的（23 回合勝率 26%、IS2 9%）。巢狀空假設檢定（confirm_bars→∞ = 通道）顯示 BTC 五格 t −1.5～−1.6、ETH 零貢獻；confirm_bars 在 BTC 的最佳值在 ∞ = 這個參數不該存在。與 20d_high 合讀：**任何降低進場門檻的 OR 腿在 BTC 都往基準退化（回撤變深、Δ 轉負），通道 +1 ATR 的「挑剔」本身就是 BTC 的 alpha**。快速重進方向關閉。
3. **relvol_ema50_downside**：第 1 輪 P3 / 方向 4 假設「相對尺度解決 IS 校準與死碼、只算下行讓噴出段不被砍」——IS 校準**解決**（sizing 四參數不去重十格 100%），死碼**沒解決**（趨勢內 dvol/ref 比值 p90 只有 1.25，因為 365 日中位數 ref 含趨勢外的崩盤日，1.4× ≈ p95；BTC 2022/2023/2025/2026、ETH 2022/2023/2025/2026 趨勢內砍倉 0 天），噴出段假設**推翻**（BTC 每一次砍倉都在牛市加速段，2021-01 −11.9%；下行半變異在拋物線上漲裡一樣飆）。BTC IS/FULL/IS2 t −2.25 是本輪唯一顯著的結果，方向為負。ETH +17.8% 與第 1 輪 downside_gate 一樣集中在 IS 少數事件。方向 4 關閉。
4. **relvol_ema50_total**：對照組隔離了「只算下行」的貢獻：假設 (b)「噴出段不砍」沒有實現（主變體滿倉、對照砍倉的 A 日實現差只 +0.4～+1.1%，因為對照的 raw target 0.91–0.94 被 0.15 門檻吸收沒成交）；下行版的真實機制是「崩盤日比值更極端所以砍得到」，只在 ETH 賺、在 BTC 賠。回答第 1 輪的問題：shock_budget 的死碼**不是絕對尺度造成的**，相對 ref 只把死碼從「2025 後」搬到「2021、2024 以外」，本質是 20 日 / 365 日中位數比值 > 1.65 的日子只佔趨勢內 3–5%。七年 3/2 次減倉成交 = 無操作變體。
5. **closestop_ema50_c3atr_hold7**：第 1 輪 P1 診斷「傷害來自下影線觸價與立即重進」**成立**——同參數 ema50_atr（BTC IS 0.23）/ chandelier_immediate（0.60）→ 0.88，ETH 0.85/0.96 → 1.57；但去除汙染後「3×ATR 停損只截斷真正崩盤」在 BTC **推翻**：BTC 沒有任何頂部高於 EMA50 + 3 ATR 再崩，17 次停損裡 11 次與 EMA50 出場同根（Δ 0）、6 次是趨勢未斷的 V 型回檔、冷卻後 100% 在高於停損價 +8.6% 處重進（比第 1 輪 immediate 版的 78% 更糟，因為「收在觸發價上 + 隔 7 根」定義上就是反彈成立後才進）。ETH IS +0.58 log 有 56% 來自 2021-05-19 一天；k 曲面 BTC 單調（越鬆越好）、ETH k=3 單一事件尖峰 = 過擬合特徵。EMA50 + 固定倍數 ATR 停損關閉。順帶找到引擎缺口（延遲 × 收盤停損）。
6. **crashgate_ret5_atr_abs**：報酬衝擊型崩盤閘在 k=3 是死碼——BTC 6.8 年 43 個觸發日只有 2 天 close 還在 EMA50 之上、ETH 45 天只 1 天：日線加密幣 5 天內跌 3 個 ATR（−13～−25%）幾乎必然已穿破 EMA50。放鬆到 k=2.25 它就變成回檔濾網（第 1 輪 exit_ema50_stall「賣在洗盤低點」重演），兩者之間沒有平台。它唯一能比 EMA50 早一天的情境是從遠高於 EMA50 的位置起跌（2021-02-26 BTC，價值 +2.5%），而同規則在 2024-03 兩標的各付一次 +3～10% 的假旗標。
7. **crashgate_dvol_ratio_abs**：第 1 輪方向 1 假設「ETH IS 1.43→1.5+、BTC 不劣化」**推翻**：價格收復解除把閘門變成幾乎空轉（旗標 48/53 次只有 3 次在趨勢內亮起、亮起後中位 1 根就被「收盤 > 旗標日收盤」解除），兩標的 in_regime 各只改 5 根，第 1 輪 downside_gate 在 ETH 的改善被整個抹掉；消融「無價格解除」才有內容（ETH 五格全過、OOS +0.14），但 BTC IS/IS2 −0.11（2021-01-27 那一次原樣重演），第 1 輪的拉鋸完整保留；絕對門檻只加 +0.02。價格收復解除機制關閉；崩盤閘要有效必須讓旗標存活到崩盤走完。
8. **crossasset_ema50_majority_vote**：5 日中值濾波把回合砍半（39/40 vs 79/83）而在場率不變，證實鋸齒是短天期事件；但 2 根出場延遲讓 OOS 回撤更深（BTC −30→−34%、ETH −51→−64%），ETH OOS/OOS2 掛在少數轉折日（2025-05-10 錯過 +10% 突破、2024-04-08/13 各 −7% 沒出、2024-11-07 錯過 −6.3%）。延遲一根反而讓 ETH OOS 過閘——時點運氣不是規則優勢。(5,3) 在 ETH 2024 後是尖點。時間域多數決關閉。
9. **crossasset_ema50_run_hysteresis**：連續 2 根遲滯砍四成翻轉、無價格尺度參數（跨標的可搬目標達成），但對稱的一根出場延遲把每次真崩盤的第一天整根吃進來（2022-11-09 FTX 單日輸基準 BTC −14%、ETH −17%），十格回撤全深 1–6 pp；進場延遲在 2023 後 V 型市場付的是動能日（BTC 2025-04-22 −6.7%、ETH 2024-11-07 −6.3%），IS 賺的與 OOS 賠的是同一個機制，說明 n=2 的正貢獻是 IS 市場結構（熊市反彈短命）而非跨期性質。第 1 輪「進場確認、出場不拖延」在時間域再次成立；對稱時間遲滯關閉。
10. **closestop_channel_c3atr_hold7**：regime_key 與通道完全一致，所以是純停損歸因：BTC 21 次停損裡 11 次與通道出場同根（所有真正崩盤腿：2021-04/11、2022-08/11、2025-08/10、2026-01/05）、10 次是趨勢中回檔，停 7 根後在高 4–13% 處重進（2020-05-12 −12.2%、2024-03-20 −11.7%），對通道期末權益比 0.75；ETH IS +44% 幾乎全來自 2021-05-18 與 2020-09-04 兩次，OOS −10%。與第 5 名合讀：**3×ATR 收盤確認停損的負增量與 regime 無關**，在 BTC 不傷的只有 ≥ 3.75 ATR，而那時它是死碼。第 1 輪方向 2「ETH OOS 回撤 −48% → −40%」推翻（實際 −52.9%，更深）。holdout 預覽 SOL 4/5、BNB 1/5，沒有跨標的一致增量。

### 8.1 跨變體的結構性結論

1. **EMA50 之上加不了「保護」**：崩盤閘（兩個）與停損（兩個）四個變體共同證明，日線 EMA50 對崩盤的反應（跌破即出，約一週尺度）已快到讓任何「更早出場」的規則只剩兩種結局：與 EMA50 同根（Δ 0）或在回檔裡出（負）。
2. **EMA50 之上加不了「放鬆」**：快速腿（兩個）證明任何 OR 進場腿在 BTC 是零期望對稱賭注，代價是通道的回撤與換手優勢。
3. **對稱的時間延遲在日線加密幣不可承受**：出場側一根 = 一整根崩盤日（−14～−17%）；進場側一根 = V 型市場的動能日（−6～−10%）。
4. **波動目標式的曝險控制在趨勢內幾乎不動作**（2–7% 的日子），動作時方向在 BTC/ETH 相反。
5. **ETH OOS 對持有的邊距（0.045 Sharpe）是整個家族的共同弱點**，基準本身在資金費率情境下 ETH 五格全掛。任何變體的「ETH OOS 過閘」都在雜訊內；取代冠軍規則要求 ETH OOS Δ ≥ −0.05 是對的，但「過閘」本身不該被當作 ETH 上的證據。
6. 兩輪 22 個變體、200 個對基準的格，沒有任何一格有 |t| ≥ 2 的正差異。`ema50_plain` 在「單標的、日線、EMA50 濾波」這個族裡是局部最優；下一步要換的是資訊集或部位表示。

---

## 9. 冠軍決定

**champion = ema50_plain（維持）。** 十個變體對取代門檻逐項檢查：

| 變體 | 十格 ≥ 9 且四 OOS 全過 | 六格平均 > 1.09 | 四 OOS Δ ≥ −0.05 | OOS 敏感度 ≥ 0.67 | lookahead / overfit / holdout |
|------|----------------------|----------------|-----------------|------------------|------------------------------|
| fastreentry_channel_or_20d_high | ✓ 10/10 | ✓ 1.155 | ✗ BTC OOS −0.098 | ✓ 1.00 | pass / **fail** / pass |
| fastreentry_channel_or_3bar_confirm | ✓ 10/10 | ✓ 1.134 | ✗ BTC OOS −0.093、OOS2 −0.065 | ✓ 0.90 | pass / **fail** / pass |
| relvol_ema50_downside | ✓ 10/10 | ✗ 1.089 | ✓ | ✓ | 自檢 pass / — / pass |
| relvol_ema50_total | ✓ 10/10 | ✗ 1.086 | ✓ | ✓ | 自檢 pass / — / — |
| closestop_ema50_c3atr_hold7 | ✗ ETH OOS2 | ✓ 1.094 | ✗ −0.157 | ✗ | — |
| crashgate_ret5_atr_abs | ✗ ETH OOS2 | ✗ 1.022 | ✗ −0.093 | ✓ 0.75 | 自檢 pass / — / — |
| crashgate_dvol_ratio_abs | ✗ ETH IS（9/10 但 IS 非 OOS）→ 形式上過 | ✗ 1.017 | ✓ −0.033 | ✓ 0.875 | — |
| crossasset_ema50_majority_vote | ✗ 8/10 | ✓ 1.117 | ✗ −0.138 | ✗ 0.333 | 自檢 pass / — / — |
| crossasset_ema50_run_hysteresis | ✗ 8/10 | ✗ 1.069 | ✗ −0.184 | ✗ 0.50 | 自檢 pass / — / — |
| closestop_channel_c3atr_hold7 | ✗ 7/10 | ✓ 1.137 | ✗ −0.271 | ✗ 0.00 | — / — / 預覽 5/10 |

最接近的兩個（快速重進）都栽在同一條：BTC OOS 對冠軍 −0.09～−0.10，且過擬合審查發現它們對自己的母體 `vol_atr_channel` 在 BTC 五格全負、正差異來自 2–4 筆交易。`vol_atr_channel` 的定位不變（BTC 回撤較小、換手較低的替代品，holdout SOL 3/5 / BNB 0/5），本輪沒有任何變體在 holdout 上同時改善 SOL 與 BNB。

---

## 10. 引擎 / 風控模型改進提案（需改 `abgt/strategies/regime.py`）

第 1 輪 P1–P4 已全部實作（收盤確認停損與重進門檻、資金費率常數與執行延遲、相對 / 下行波動目標、引擎狀態機與回撤暫停解除）。以下是本輪暴露的新缺口。

### P1. 修正 `stop_trigger="close"` × `exec_delay_bars > 0` 的重進繞過

- 現況：第 i 根收盤觸發停損後，出場單排在 i+1+delay 成交；第 i+1 根 `pos` 仍 > 0，`want = on and not paused and (pos > 0 or can_enter)` 因 `pos > 0` 為真，`ref = pending[-1][1] = 0`，於是 `target > 0 and ref == 0` 又排一筆「進場」到 i+2+delay，繞過 `wait_price` / `wait_reset` 與 `reentry_min_bars`。兩個 closestop 變體的延遲一根壓力列因此無效（BTC 7/17、ETH 9/17 個停損在出場後 1 根就重進，停損計數 27/25）。
- 修法：加狀態 `exiting`（pending 中存在 target = 0 的單即為 True）；`want = on and not paused and ((pos > 0 and not exiting) or can_enter)`；停損決策根到成交根之間 `can_enter` 一律 False（不論 `reentry` 模式）；`stop_bar` 改記成交根、`reentry_min_bars` 從成交根起算；`mode` 的轉移在成交時套用，reset / immediate / close_above_stop 語意不變。趨勢結束（非停損）排隊中的處理同樣套用 `exiting`，避免延遲 > 1 時「出場還沒成交又排進場」。
- 同時 `extra` 加 `stop_events: list[dict]`，每次停損記 `{decision_bar, fill_bar, fill_px, hi_close, stop_level, reentry_bar, reentry_px, path}`（path ∈ reset / price / immediate / regime_off）。
- 預期：停損類變體的延遲情境可採信；驗證者不必重寫停損逐次歸因腳本（本輪兩位實作者各寫了一份）。

### P2. 策略可提供分級曝險與分腿出場：`in_regime` dict 加 `scale`、`leg` 與多組 `exit`

- `scale: np.ndarray[float]`：每根 0–1（NaN 視為 1），`target = max_exposure × scale[i] × 波動目標因子`，經 `rebalance_threshold` 才成交；`n_entries` 只在 0 → > 0 時計、`n_exits` 只在 > 0 → 0 時計；`extra` 加 `avg_scale_in_regime`、`partial_days`。
- `leg: np.ndarray[int]`（0 = 不進、k ≥ 1 = 由第 k 腿進場）搭配 `exit: list[np.ndarray]`（第 k 腿的出場條件）；引擎在進場時記 `leg_id`，趨勢內用 `exit[leg_id][i]` 判斷出場；`extra` 加 `entries_by_leg`、`exits_by_leg`。舊語意（單一布林陣列、`dict(enter, exit)`）不變。
- 動機：本輪快速腿失敗的結構原因是「與通道腿共用出場」把它變成對稱賭注；分腿出場讓未確認的進場可以用較緊出場（例如 close < EMA50）而通道腿維持 −0.25 ATR；分級曝險讓鋸齒付一半成本、趨勢確立後才滿倉。目前變體只能給布林，曝險只能由引擎的波動目標決定，這兩個想法在現行引擎無法測試。
- 預期：第 11 節方向 5（graded_exposure）的前提；也讓「EMA 長度分散」這種直接針對 ema_len 脆弱性的設計成為可能。

### P3. 引擎輸出 `exposure_key` / `fill_key`，讓 RiskParams 參數的敏感度不被 regime_key 去重吞掉

- `simulate_regime` 在 `extra` 加 `exposure_key = md5(round(exposure[start_idx:], 4))` 與 `fill_key = md5(成交根序列 + 方向 + 原因)`；評估器的敏感度去重改成「regime_key 相同**且** exposure_key 相同」才判 same_regime。順帶加 `n_fills`（含調倉）。
- 動機：本輪 relvol ×2、closestop_ema50、closestop_channel 四個變體的 n_effective 被算成 2（只剩 ema_len），停損倍數、冷卻根數、vol_target 的擾動全被剔除；summary 印出的 0% / 100% 都不是真的（closestop_ema50 全 8 列 BTC OOS 其實 62.5%），實作者只好自己另算。
- 預期：風控類變體的敏感度統計恢復意義；不改引擎語意，只多兩個雜湊。

### P4. 歷史資金費率逐日陣列取代常數（`funding: np.ndarray | None`），由 `data.py` 抓取

- `RiskParams` 加 `funding: np.ndarray | None = None`（每根日線的費率合計，例如當日三次 8h 費率之和；有值時覆蓋 `funding_rate_8h × 3`），結算點不變（收盤後、算 dd 前，`cost = pos × c[i] × funding[i]`，允許負費率為收入）。`data.py` 加 `fetch_funding(symbol)`（ccxt `fetchFundingRateHistory`，CSV 快取、之後只補新）。`evaluate_variant --stress` 加第五組「歷史資金費率」，報表印該情境的年化資金成本。
- 動機：現行 0.01%/8h 常數 ≈ 做多期間每年 11%，是中性費率的上限式假設；本輪 8/10 個變體與基準本身在此情境下 ETH 全掛，這一列的資訊量為零。每位可執行審查者都提出「只跑現貨」，但沒有人能回答永續到底差多少。
- 預期：回答「永續執行是否可行」而不是「最壞情況會不會翻」；也是運行閘門成本模型的依據（CLAUDE.md 路線圖既有項目）。

---

## 11. 下一輪值得試的方向

避開已證明無效的（兩輪合計）：時間停滯出場、low 觸價停損 + 立即重進、收盤確認固定倍數 ATR 停損（任何 regime 上）、月頻判定、均線對均線交叉、家族多數決、Donchian 多尺度多數決、EMA 階梯投票、絕對 / 相對波動目標（total / downside）、報酬衝擊崩盤閘、下行比值崩盤閘 + 價格收復解除、與通道共用出場的快速進場 OR 腿（20 日新高 / N 根確認）、對稱時間域遲滯（多數決 / 連續根）。

共通規則（每個方向都適用）：參數只准用 IS 與 holdout SOL/BNB 挑，BTC/ETH OOS 不得回調；報表必列對基準與對 `vol_atr_channel` 的逐格 Δ 與 t；延遲一根的四個 OOS 格至少三格過才算可上線候選。

1. **context_adaptive_entry — 進場門檻依情境在 plain 與 channel 之間切換**。第 2 輪證明降低門檻的 OR 腿在 BTC 是零期望、通道 +1 ATR 在 BTC 是 alpha，但通道在 2023 後 V 型市場付 1 ATR 延遲。要的是能區分「崩盤後首次回升」與「熊市第 N 次反彈 / 盤整鋸齒」的情境變數。候選：(a) 自身鞭打計數——過去 60 根內「持有 ≤ 5 根且虧損」的回合 ≥ 2 → 切到 +1 ATR 門檻，直到出現一次持有 ≥ 20 根的回合才切回 plain；(b) 出場冷卻——上次 regime 出場後 5 根內只准以 +1 ATR 重進；(c) 熊市情境——close < EMA200 或距 200 日高 ≥ 30% 時用 +1 ATR，否則 plain。出場固定 close < EMA50 − 0.25 ATR（另跑 close < EMA50 版）。假設：BTC IS ≥ 1.15、BTC OOS 對基準 Δ ≥ −0.03、ETH 十格 Δ ≥ 0、BTC IS 回撤 ≤ −52%。判準：十格全過、四 OOS Δ ≥ −0.05；報告情境切換次數與每種情境下的回合勝率 / 對數合計。若 (a)(b)(c) 都無法讓 BTC OOS Δ ≥ −0.05，結論「進場門檻的情境切換」關閉。
2. **btc_leader_gate — 用 BTC 的趨勢狀態做其他標的的閘門（真正的跨標的資訊）**。規則 A：ETH（及 holdout SOL/BNB）in_regime = (自身 close > EMA50) & (BTC close > BTC EMA50)，BTC 本身不變（BTC 五格與基準逐格相同、Δ = 0）。規則 B（對照）：ETH in_regime = (ETH close > EMA50) & (ETH/BTC 比值 > 其 EMA50)。實作：build 內用 `abgt.data` 讀 BTC 日線快取，以 df.index 對齊並**截到 df.index[-1]**（截斷測試時 BTC 序列必須一起截斷），缺值視為 False。假設：崩盤由 BTC 帶動、ETH 的 2024-04 / 2025-02 假訊號 BTC 同步或更早出場，ETH OOS 0.71 → ≥ 0.78、OOS2 0.45 → ≥ 0.52、回撤 −51% → ≤ −46%；風險是 BTC 先出而 ETH 續漲（2021-05 前 ETH 領漲）讓 IS 掉。判準：因 BTC 沒貢獻，取代冠軍要 ETH 五格全過、ETH OOS/OOS2 Δ ≥ +0.03、至少一格 |t| ≥ 1、holdout SOL/BNB 同規則各 ≥ 4/5；列印 ETH 與基準 in_regime 不同的根數與那些日子的相對報酬。規則 A 對 ETH OOS 為負就直接結案，不調 EMA 長度救。
3. **slow_vol_scaled_band — 通道帶寬改用慢速波動尺度**。現行 ATR14 在崩盤後膨脹讓 +1 ATR 在 V 型反轉時是 6–10% 的距離（OOS 延遲來源），盤整低波動時縮到 2–3%（鋸齒來源）。變體：band = median(ATR14/close 過去 365 根) × close，進場 close > EMA50 + 1.0 × band、出場 close < EMA50 − 0.25 × band（無量綱、跨標的語意相同）；對照用 ATR50。假設：V 型反轉進場提早 2–5 根（BTC 2023-01、ETH 2023-10）、低波動盤整（BTC 2024-05～09）門檻反而更高；BTC OOS 對基準 Δ ≥ −0.03、BTC IS ≥ 1.20、回撤與通道同級。判準：十格全過、四 OOS Δ ≥ −0.05、對 vol_atr_channel BTC 至少三格 ≥ 0、k_in/k_out ±25% OOS 通過率 ≥ 0.75、延遲一根 BTC OOS2 要過（通道過、快速腿版不過）；報告進場日相對通道提早 / 延後的根數分布與多餘回合數。
4. **volume_confirmed_entry — 成交量當進場確認（至今沒用過的資訊源，同根判定、零延遲、無量綱）**。enter = close > EMA50 且 vol_ratio ≥ q，vol_ratio = 當根 volume / 過去 20 根 volume 中位數（不含當根），q = 1.0（對照 1.25）；exit = close < EMA50（另跑 −0.25 ATR 版）；穿越當根量不足時之後任一根滿足即進。先檢查 2019–2020 早期 volume 是否有零 / 異常，異常則從 2020-07 起算並註明。假設：熊市反彈與盤整鋸齒的穿越發生在量能低於中位數的日子（BTC 2022-02/07/09、2024-05/07/08 那些快速腿失敗日），V 型突破日量能 ≥ 1.5×；預期 BTC IS ≥ 1.12、BTC OOS Δ ≥ −0.03、回合降到 50–60、ETH 至少持平。判準：十格全過、四 OOS Δ ≥ −0.05、q ±25% 與量窗 15/25 的 OOS 通過率 ≥ 0.67、延遲一根 BTC OOS 要過；必列「量能過濾掉的穿越日」名單與那些日子若進場的報酬——若濾掉的日子平均報酬為正，方向關閉。
5. **graded_exposure — 二元進出換成分級曝險（依賴 P2 `scale` 陣列；引擎未升級則延後）**。(a) 距離分級：目標曝險 = clip((close − EMA50)/ATR14, 0.5, 1.0)，close < EMA50 → 0；即剛站上 EMA50 時 50%、到 +1 ATR 滿倉，出場 close < EMA50 − 0.25 ATR 全出（另跑 close < EMA50 版）。(b) 長度分散：曝險 = mean(close > EMA_n for n in {35, 50, 70})，得 0/⅓/⅔/1 四級，直接攻擊「ema_len 是唯一脆弱參數」的問題。rebalance_threshold 0.15。假設：(a) 鞭打成本減半、V 型首日只損失一半曝險、BTC IS 回撤 −56% → ≤ −50%、BTC OOS Δ ≥ −0.03；(b) ema_len ±25% 十格通過率 100% 且 OOS Sharpe 帶寬縮到 ±0.05。判準：十格全過、四 OOS Δ ≥ −0.05、成交次數 ≤ 基準 1.5 倍、費 0.10%+滑 0.20% 十格過；報告平均曝險（預期 45–50%）與「曝險介於 0 與 1 之間的日子」的相對報酬合計。

若第 3 輪仍無冠軍，依規則停止迴圈；建議「停止」的具體內容是：`ema50_plain` 做 Pine 版與紙上前向、把資源轉到執行面（現貨 vs 永續的資金費率實測、運行閘門、Binance 測試網），而不是再開第 4 輪 EMA50 濾波變體。

---

## 12. 對評估器 / 迴圈流程的建議（不改 regime.py）

1. **ETH 視窗對齊**：ETH 快取從 2019-11-27 起，warmup > 49 的變體 ETH IS/FULL 起點會後移，評估器印的對基準 Δ 變成跨視窗數字（本輪兩個崩盤閘 ETH IS Δ −0.21 / −0.27，真值 0.000 / −0.034）。改法：對基準以變體的 warmup 重跑再算 Δ，或在報表標記「視窗不同」並印同視窗 Δ。
2. **holdout 判準按標的分開**：合併平均讓 SOL +0.14 遮掉 BNB −0.11（20d_high 形式上過）。建議各標的平均 Δ ≥ −0.05，且冠軍過閘的格不得翻成掛。
3. **母體（巢狀特例）自動並列**：對「修改自某父變體」的變體，報表自動附父變體的逐格 Δ 與 t；本輪對基準 t 0.4–1.0 看似中性，對父變體 t −1.5 才顯示真相。
4. **「enter 是母體超集、exit 相同」的標準診斷**：印額外持倉日數、拆成「提早進場」與「多餘回合」兩類的逐段損益（按次日開盤成交）；設計摘要用的 fast_sum_ret 把整段後續行情記給快速腿，高估約 10 倍。
5. **壓力報表附基準同情境列**：本輪三位實作者各自重跑基準壓力才分得出「遺傳」與「造成」。
6. **新參數的單調性檢查**：曲面若在參數 → ∞ 或 → 0 時最好，代表該參數應移除（confirm_bars、atr_stop_mult）。
7. **前 k 筆交易貢獻佔比**：快速腿 BTC/ETH 扣前兩筆各為 −0.54 / +0.21 log；停損 ETH IS 單日 56%。
8. **停損類變體在 P1 修正前，延遲一根壓力列標記為無效**。
9. **多代理共用 `results/holdout` 時輸出檔名要獨立**（本輪一位驗證者覆寫了評估者產物，已還原）。

---

## 13. 本輪產出檔案

- 變體：`experiments/variants/{fastreentry_channel_or_20d_high, fastreentry_channel_or_3bar_confirm, relvol_ema50_downside, relvol_ema50_total, closestop_ema50_c3atr_hold7, closestop_channel_c3atr_hold7, crashgate_ret5_atr_abs, crashgate_dvol_ratio_abs, crossasset_ema50_majority_vote, crossasset_ema50_run_hysteresis}.py`
- 結果：`results/variants/<id>.json` 與 `.txt`；holdout：`results/holdout/{fastreentry_channel_or_20d_high, fastreentry_channel_or_3bar_confirm, relvol_ema50_downside, ema50_plain, vol_atr_channel}.{json,txt}`、`results/holdout/verify_fastreentry_*.json`
- 本報告：`experiments/rounds/round_2.md`
- 迴圈狀態：`experiments/rounds/loop_state.json`（由編排端更新：champion 維持 ema50_plain、ledger 加 10 筆、CLAUDE.md 第 5 節加第 2 輪摘要）
