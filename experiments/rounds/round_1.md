# ABGT 優化迴圈 — 第 1 輪回合報告

日期：2026-09-23　｜　基準：`ema50_plain`（六格平均 Sharpe 1.09）　｜　本輪變體：12 個　｜　驗證：前 4 名各做 lookahead / overfit / executable 三道審查

## 0. 結論一句話

**冠軍換人：`vol_atr_channel`**（EMA50 ± ATR 通道遲滯，進場 +1.0 ATR、出場 −0.25 ATR）。六格全過、平均 Sharpe 1.19 > 1.09、無前視、過擬合審查 pass、可執行審查 pass。但要正確定位它：**OOS 的風險調整報酬與基準持平（BTC 1.34 vs 1.40、ETH 0.71 vs 0.71），真正的改善是回撤少 6–15 pp、換手降到基準的 1/2.5、IS 全面提升**，不是「Sharpe 更高的策略」。上線前提見第 7 節。

---

## 1. 評估設定

- 標的：BTC/USDT:USDT、ETH/USDT:USDT 日線（Binance USDT-M 永續，未計資金費率）
- 區間：IS 2020-01→2022-12（含 2021 牛市、2022 熊市）、OOS 2023-01→2026-09-22、FULL 2020-01→今
- 引擎：只做多、收盤判定、下一根開盤成交、0.05%/邊手續費、0.02% 滑點
- 閘門（風險調整）：Sharpe > 同期持有 **且** Calmar > 持有 **且** 最大回撤 < 持有，六格各自判定
- 敏感度：每個數值參數 ±25%，回報各格通過率
- 同期持有 Sharpe（暖機 49 根、與基準同視窗）：BTC IS 0.76 / OOS 1.18 / FULL 0.92；ETH IS 1.18 / OOS 0.67 / FULL 0.93

## 2. 總表：變體 × 六格（Sharpe / 最大回撤 / 閘門）

排序規則：閘門數 → 平均 Sharpe。✓ = 該格過閘門。

| # | 變體 | 閘門 | 平均 Sharpe | BTC IS | BTC OOS | BTC FULL | ETH IS | ETH OOS | ETH FULL |
|---|------|------|------------|--------|---------|----------|--------|---------|----------|
| — | **ema50_plain（基準）** | 6/6 | 1.09 | 1.01 / −56% ✓ | 1.40 / −30% ✓ | 1.17 / −56% ✓ | 1.26 / −61% ✓ | 0.71 / −51% ✓ | 0.99 / −61% ✓ |
| 1 | **vol_atr_channel** | 6/6 | **1.19** | 1.28 / −46% ✓ | 1.34 / −24% ✓ | 1.29 / −46% ✓ | 1.43 / −46% ✓ | 0.71 / −48% ✓ | 1.09 / −48% ✓ |
| 2 | vol_downside_gate | 6/6 | 1.18 | 0.91 / −56% ✓ | 1.37 / −28% ✓ | 1.11 / −56% ✓ | 1.57 / −48% ✓ | 0.88 / −42% ✓ | 1.23 / −48% ✓ |
| 3 | vol_shock_budget | 6/6 | 1.15 | 1.08 / −54% ✓ | 1.37 / −30% ✓ | 1.21 / −54% ✓ | 1.46 / −58% ✓ | 0.68 / −52% ✓ | 1.08 / −58% ✓ |
| 4 | ensemble_ema_ladder_vote | 6/6 | 1.11 | 1.12 / −59% ✓ | 1.25 / −32% ✓ | 1.17 / −59% ✓ | 1.29 / −69% ✓ | 0.78 / −50% ✓ | 1.04 / −69% ✓ |
| 5 | exit_ema50_stall | 6/6 | 1.06 | 0.89 / −56% ✓ | 1.35 / −36% ✓ | 1.09 / −56% ✓ | 1.28 / −61% ✓ | 0.73 / −51% ✓ | 1.01 / −61% ✓ |
| 6 | exit_ema50_band | 5/6 | 1.20 | 1.19 / −54% ✓ | 1.15 / −36% ✗ | 1.15 / −54% ✓ | 1.65 / −42% ✓ | 0.80 / −46% ✓ | 1.24 / −46% ✓ |
| 7 | ensemble_donchian_ladder | 4/6 | 1.08 | 1.02 / −39% ✓ | 1.16 / −28% ✗ | 1.07 / −39% ✓ | 1.52 / −50% ✓ | 0.61 / −41% ✗ | 1.10 / −54% ✓ |
| 8 | classic_tsmom_1_3_12 ¹ | 4/6 | 0.95 | 1.22 / −51% ✓ | 1.21 / −28% ✓ | 1.19 / −53% ✓ | 1.24 / −57% ✓ | 0.18 / −72% ✗ | 0.67 / −72% ✗ |
| 9 | ensemble_family_vote ¹ | 3/6 | 1.00 | 1.18 / −39% ✓ | 1.10 / −29% ✗ | 1.12 / −39% ✓ | 1.04 / −61% ✗ | 0.70 / −41% ✓ | 0.86 / −61% ✗ |
| 10 | classic_faber_10m_sma ¹ | 3/6 | 0.97 | 1.42 / −53% ✓ | 1.15 / −32% ✗ | 1.26 / −53% ✓ | 1.28 / −57% ✓ | 0.06 / −72% ✗ | 0.68 / −72% ✗ |
| 11 | classic_golden_cross_50_200 ¹ | 3/6 | 0.92 | 1.13 / −53% ✓ | 0.69 / −40% ✗ | 0.90 / −57% ✗ | 1.51 / −57% ✓ | 0.32 / −62% ✗ | 0.93 / −62% ✓ |
| 12 | exit_ema50_chandelier_immediate | 2/6 | 0.88 | 0.60 / −55% ✗ | 1.28 / −31% ✓ | 0.90 / −55% ✗ | 0.96 / −60% ✗ | 0.72 / −56% ✓ | 0.83 / −60% ✗ |

¹ 暖機較長（TSMOM 2020-09/11、Faber 2020-06/08、family_vote 90 根、golden cross 199 根），IS 起點比基準晚，IS 格與基準不是同一視窗；閘門判定本身是同視窗比較，仍有效。

### 2.1 冠軍 vs 基準逐格（Sharpe 差、回撤差）

| 格 | 基準 | 冠軍 | ΔSharpe | 回撤 基準→冠軍 | 淨報酬 基準→冠軍 | 持有淨報酬 |
|----|------|------|---------|---------------|-----------------|-----------|
| BTC IS | 1.01 | 1.28 | **+0.27** | −56% → −46% | +211% → +343% | +130% |
| BTC OOS | 1.40 | 1.34 | −0.06 | −30% → −24% | +387% → +323% | +419% |
| BTC FULL | 1.17 | 1.29 | +0.12 | −56% → −46% | +1414% → +1773% | +1097% |
| ETH IS | 1.26 | 1.43 | **+0.17** | −61% → −46% | +570% → +814% | +618% |
| ETH OOS | 0.71 | 0.71 | 0.00 | −51% → −48% | +122% → +119% | +129% |
| ETH FULL | 0.99 | 1.09 | +0.10 | −61% → −48% | +1387% → +1904% | +1552% |

讀法：Sharpe 改善集中在 IS（含 2021 橫盤與 2022 熊市），OOS 持平；回撤六格全部更淺；OOS 淨報酬兩個標的都輸持有（進場晚 1 ATR 的代價），只靠風險調整過閘。

## 3. 敏感度（±25%，各格通過率）

| 變體 | BTC IS | BTC OOS | BTC FULL | ETH IS | ETH OOS | ETH FULL | 最低 | 失敗方向 |
|------|--------|---------|----------|--------|---------|----------|------|---------|
| vol_atr_channel | 100% | 88% | 100% | 100% | 75% | 100% | 0.75 | ema_len=38（兩標的 OOS）、k_in=0.75（ETH OOS）：更快/更鬆進場在盤整市被掃 |
| vol_downside_gate | 100% | 75% | 100% | 100% | 100% | 100% | 0.75 | vol_window=15、ratio_on=0.75（BTC OOS）：旗標亮太久錯過續漲 |
| vol_shock_budget | 100% | 100% | 100% | 100% | 88% | 100% | 0.88 | vol_lookback=8（ETH OOS 0.653 < 持有 0.665） |
| ensemble_ema_ladder_vote | 100% | 75% | 100% | 75% | 75% | 75% | 0.75 | 任何讓 k_min 從 2 變 3 的擾動（n_rungs=5、vote_frac=0.625）ETH 全掛；base_len 19/31 BTC OOS 掛 |
| exit_ema50_stall | 100% | 88% | 100% | 75% | 88% | 100% | 0.75 | stall_bars=15 兩標的 OOS 都掛（更多觸發 = 更多鞭打） |
| exit_ema50_band | 100% | 67% | 100% | 100% | 83% | 100% | 0.67 | ema_len=62、atr_len=18（BTC OOS 差 0.03）；band_atr=0.625 ETH OOS 跳水到 0.61 |
| ensemble_donchian_ladder | 100% | 0% | 100% | 100% | 50% | 100% | 0.00 | BTC OOS 8 組全部差持有一點；ETH OOS 過的 4 組沒有一致方向 |
| classic_tsmom_1_3_12 | 100% | 50% | 75% | 100% | 25% | 25% | 0.25 | lb_long=9 BTC 最好 / ETH 最差：兩標的對回看期偏好相反 |
| ensemble_family_vote | 100% | 18% | 100% | 32% | 82% | 64% | 0.18 | k_min 是唯一決定性參數，沒有一個 k 同時穩 |
| classic_faber_10m_sma | 100% | 0% | 50% | 100% | 0% | 0% | 0.00 | 月頻節奏問題，不是參數問題 |
| classic_golden_cross_50_200 | 100% | 0% | 25% | 100% | 0% | 50% | 0.00 | 均線 vs 均線的滯後結構性 |
| exit_ema50_chandelier_immediate | 67% | 67% | 83% | 0% | 67% | 17% | 0.00 | ETH IS 6 組 0% 通過，結構性失敗 |

補充（來自驗證者的 ±50% 與替換切分，只做了前 4 名）：

| 變體 | 替換切分 IS 2020-07→2023-12 / OOS 2024-01→ | ±50% 擾動 | 判讀 |
|------|------------------------------------------|----------|------|
| vol_atr_channel | 6/6 過；BTC OOS 0.94（持有 0.75）、ETH OOS 0.55（持有 0.42） | k_in/k_out 16 次全過；所有 OOS 失敗都在 ema_len（25 四情境全掛） | 超額 Sharpe 對持有跨切分穩定（BTC +0.16→+0.19、ETH +0.05→+0.13）；k_in×k_out 42 格網格是寬平台 |
| vol_downside_gate | 6/6 過；BTC OOS 0.89、ETH OOS 0.71 | 6/8 組過 5–6 格；ratio_off=0.75、ratio_on=0.5 是退化設定（反向遲滯） | 網格寬平台，但 BTC 12 格（兩切分）全部略輸基準 |
| vol_shock_budget | **ETH OOS 失敗**（0.403 < 持有 0.422），該切分 ±25% ETH OOS 通過率 37.5% | 原切分 ETH OOS 6/8；替換切分 3/8、最低 0.27 | vol_target 0.70 是 IS 校準；BTC 2025 後預算成死碼 |
| ensemble_ema_ladder_vote | 6/6 過，但 OOS Sharpe 掉幅 > 0.3（BTC 1.25→0.91、ETH 0.78→0.45） | BTC OOS 50%、ETH IS 38%；k_min=3 方向全敗、k_min=1 方向 ETH OOS 反而更好 | 對基準日報酬差 t 值 < 0.5；4 參數換 1 參數沒換到東西 |

對照：基準 `ema50_plain` 在替換切分下 BTC 1.32 / 0.97 / 1.19、ETH 1.22 / 0.45 / 0.95，六格全過，±25% 與 ±50% OOS 通過率 100%。

## 4. 驗證結論（前 4 名）

| 變體 | lookahead | overfit | executable | 關鍵理由 |
|------|-----------|---------|------------|---------|
| vol_atr_channel | **pass**（none） | **pass**（low） | **pass**（medium） | 純 Python 重算 EMA/ATR/狀態機 0 筆不一致；203/227 個切點截斷測試 0 筆不一致；替換切分 6/6、超額 Sharpe 穩定、k_in×k_out 寬平台。可執行：一年 8–10 邊、成本 0.6–0.7%/年，費 0.10%+滑 0.05% 六格仍過。疑慮：每視窗回合數只 11–17；ETH OOS 邊距 0.05；獲利集中（前 5 筆佔毛利 68–78%）；資料是永續但未算資金費率（0.01%/8h 下 BTC OOS 會掉到 1.17 < 1.18） |
| vol_downside_gate | **pass**（none） | **pass**（medium） | **fail**（medium） | 無前視、網格寬平台。但 BTC 三區間全輸基準（改善 100% 來自 ETH）；ETH +121% 相對報酬集中在前 10 個單日（+147%）、執行延遲 1 根後 ETH 回撤從 −48% 退回 −64%；BTC 2021-01-27 單次邊緣觸發（ratio 1.54 vs 門檻 1.5）錯過 +52.9%；兩標的參數曲面沒有共同高原 |
| vol_shock_budget | **pass**（none） | **fail**（medium） | **fail**（medium） | 替換切分 ETH OOS 閘門失敗、該切分敏感度 37.5%；改善全在 IS 的兩次波動爆發（2020-03、2021-05），2023 後所有 OOS 格持平到略差；vol_target 0.70 = IS rvol p65，BTC 2025–2026 預算 0 天生效；ETH OOS Sharpe 在 0.70 恰是最低點 |
| ensemble_ema_ladder_vote | **pass**（low） | **fail**（medium） | **pass**（medium） | 無前視、暖機期不影響閘門。但替換切分 OOS 掉 > 0.3、ETH 超額 Sharpe 對持有從 +0.12 縮到 +0.03；對基準日報酬差 |t| < 1；六格中五格回撤更深（ETH 深 7.5 pp）；多頭排列「容忍到 EMA100」BTC 為負、ETH 為正互相否定 |

三位驗證者共同指出的評估器問題（第 8 節有對應建議）：「替換切分後 OOS Sharpe 掉 > 0.3 判 fail」量到的是 2024 年後市場的 Sharpe 水準（持有本身 −0.43、基準 −0.43），應改成超額 Sharpe（對持有或對基準）的變化。

## 5. 換手與曝險

| 變體 | BTC 回合（FULL） | BTC 在場 | ETH 回合（FULL） | ETH 在場 | 停損 / 暫停 |
|------|-----------------|---------|-----------------|---------|-----------|
| ema50_plain | 79 | 54% | 83 | 54% | 0 / 0 |
| vol_atr_channel | **28** | 48% | **34** | 50% | 0 / 0 |
| vol_downside_gate | 72 | 51% | 59 | 49% | 0 / 0 |
| vol_shock_budget | 79（成交 106） | 54% | 83（成交 157） | 54% | 0 / 0 |
| ensemble_ema_ladder_vote | 67 | 62% | 71 | 62% | 0 / 0 |
| exit_ema50_stall | 84 | 53% | 86 | 53% | 0 / 0 |
| exit_ema50_band | 37 | 55% | 32 | 55% | 0 / 0 |
| ensemble_donchian_ladder | 24 | 40% | 23 | 40% | 0 / 0 |
| classic_tsmom_1_3_12 | 8 | 57% | 10 | 63% | 0 / 0 |
| ensemble_family_vote | 49 | 48% | 56 | 47% | 0 / 0 |
| classic_faber_10m_sma | 2 | 71% | 6 | 57% | 0 / 0 |
| classic_golden_cross_50_200 | 6 | 58% | 6 | 57% | 0 / 0 |
| exit_ema50_chandelier_immediate | 101 | 53% | 102 | 53% | 32–33 / 0 |

觀察：換手最低的四個（Faber、黃金交叉、TSMOM、Donchian 階梯）OOS 全部失敗——在加密貨幣 2023 年後的 V 型市場裡，「慢很多的訊號」換掉的是手續費，換來的是轉折點延遲 2–4 週的機會成本，後者遠大於前者。冠軍把換手從每年 12 回合砍到 4–5 回合而 OOS 不掉，是靠「進場要確認、出場不拖延」的不對稱遲滯，不是靠拉長回看期。

## 6. 每個變體教了我們什麼（排名順序）

1. **vol_atr_channel**：不對稱 ATR 通道遲滯（進場 +1.0、出場 −0.25）是目前唯一在六格全過、換手砍到 1/2.5、回撤六格全淺且通過三道驗證的規則；對稱版（exit_ema50_band ±0.5）BTC OOS 掛，證明「進場確認、出場不拖延」才是關鍵，不是帶寬本身。
2. **vol_downside_gate**：下行/上行半變異比值的崩盤閘對 ETH（波動大、崩盤深）是大正貢獻（IS 1.26→1.57、回撤 −61%→−48%），對 BTC 恆為小負貢獻（2021-01 邊緣觸發錯過 +53%）；無量綱比值在平靜市場也會觸發，且解除條件靠 20 天窗滾出，V 型反彈必落後。
3. **vol_shock_budget**：把波動目標從停損/重置隔離出來後證實它「單獨」不傷（先前 ema50_voltarget 的傷害來自停損+重置），但絕對門檻 0.70 是 IS 校準、BTC 2025 後成死碼、ETH 替換切分掛，波動目標要做必須尺度無關。
4. **ensemble_ema_ladder_vote**：K-of-N 對巢狀 EMA 等價於「close > 第 K 低的 EMA」，自動在牛市放寬出場到 EMA100；淨報酬 5/6 格更高但回撤 5/6 格更深、對基準 t 值 < 0.5，是橫向替代不是升級。
5. **exit_ema50_stall**：時間停滯在加密貨幣標記的是盤整低點而不是頂部（5/8 次鞭打，賣在洗盤低點、買在 +4–9% 的突破），而真正傷人的頂部都是快速崩盤，EMA50 一週內就出，時間規則永遠到不了觸發點。
6. **exit_ema50_band**：Schmitt 觸發把鋸齒砍一半、IS 兩標的大幅改善、ETH 三格全升，但對稱帶在低波動單邊行情（BTC 2023–24）每邊各付 1–1.5% 延遲成本，差 0.03 掛 BTC OOS；它是冠軍的種子。
7. **ensemble_donchian_ladder**：多尺度突破多數決在 2022 熊市曝險只 5–7%、IS 回撤 −39%/−50%，但 100 日機關閉時退化成「55 日新高進、10 日新低出」= 買高被掃，兩個 OOS 都差持有一點；尺度分散帶來的是 IS 躲熊能力，不是 OOS 優勢。
8. **classic_tsmom_1_3_12**：月末判定的 TSMOM 在 BTC 大週期成立（2022 全年空手），但只做多、無日內止血下，ETH 單月 −20～−30% 的急跌把 Calmar 與回撤同時擊穿（OOS −72% 比持有還深）；月度粒度視為已探完。
9. **ensemble_family_vote**：五家族多數決確實把 BTC 熊市回撤從 −56% 壓到 −39%，但 k_min=3 剛好卡在橫盤鞭打區、進場門檻高錯過 2023 初與 2024-09 後的反彈、ETH 急崩不比 EMA50 快；沒有一個 k 同時在牛熊與橫盤都穩。
10. **classic_faber_10m_sma**：10 月均線在 BTC 是理想的躲熊器（IS +405% / Sharpe 1.42），但 ETH 2023 後 2–3 個月一波的 V 型震盪比月頻訊號還快，四個 OOS 回合全部買在前波高、賣在前波低。
11. **classic_golden_cross_50_200**：均線對均線的滯後在 2023 後系統性「底部出、頂部進」（2025-04 死叉 79k → 05-22 金叉 112k），四組擾動沒有一組讓 OOS 過；失敗來自結構不是 50/200 這組數字。
12. **exit_ema50_chandelier_immediate**：觸價（low）停損 + 立即重進在日線加密幣等於付費買下影線保險——78% 的重進價高於停損價、IS 累計溢價 +65%（BTC）/ +114%（ETH）；ema50_atr（reset 版）的傷害有一半來自「錯過續漲」、另一半來自低賣高買本身。停損要有用，出場判定必須改收盤確認、重進要有價格門檻。

## 7. 冠軍決定與上線前提

**champion = vol_atr_channel**。符合本輪規則的四項條件：六格全過、平均 Sharpe 1.19 > 1.09、lookahead pass、overfit pass（可執行審查也 pass）。

但收錄時的定位與條件（三位驗證者共識）：

1. **定位**：「OOS 風險調整報酬 ≈ 基準、回撤少 6–15 pp、換手 1/2.5、成本拖累 0.6–0.7%/年」，不是「Sharpe 更高」。六格中 ETH OOS 為邊際通過（0.71 vs 持有 0.665），優勢集中在 IS 2020–2022。
2. **標的**：先只上 BTC。ETH OOS 邊距在雜訊內，任何額外摩擦（滑點 0.2%、資金費率 0.005%/8h）都會翻盤；ETH 目前處於 −47.7% 未回復回撤、2024–2026 16 筆裡 12 筆虧損。
3. **場所**：現貨（無資金費率），不在 USDT-M 永續跑；若要永續，先把歷史資金費率併進引擎重跑（第 8 節 P2）。
4. **初始化**：狀態機有路徑依賴，從不同日期起算要 48–140 天才與全歷史版本一致；實盤必須從快取起點（2019-09）重放完整 EMA50/ATR14/狀態機，上線前印出當日 upper/lower/state 與回測最後一根比對。
5. **選擇偏誤**：本輪已在同一 OOS 上比較 16 個變體，且冠軍是從近親 exit_ema50_band 的 OOS 失敗中改出來的，這個 OOS 已被用來選擇。上線前補真正沒碰過的 holdout（SOL/BNB 日線同參數）+ 至少兩季紙上前向。
6. **參數凍結**：ema_len 是唯一脆弱參數（±50% 兩側都可能掛 BTC OOS），固定 50 視為結構常數；atr_len ±25% 對 BTC in_regime 完全無影響（沒有鑑別力），穩健性判讀以 k_in、k_out 為主。
7. **運行閘門**：本變體在強牛年（2023、2024）會落後持有 15–50 pp，運行閘門的「落後 X% 轉持有」要先定義可接受的落後幅度，否則會在 2023 那種年份被人為停掉。

## 8. 引擎 / 風控模型改進提案（需改 `abgt/strategies/regime.py`）

### P1. 吊燈停損改收盤確認 + 重進門檻

- 新欄位：`stop_trigger: str = "low"`（`"low"` 保留現行觸價語意；`"close"` = 第 i 根收盤 `c[i] < stop` 才觸發，併入 pending 於 i+1 開盤成交、含滑點，不再用 `min(o[i], stop)` 的 K 線內成交價）。
- `reentry` 新增第三種值 `"close_above_stop"`：停損後記住 `stop_exit_px`，只有 `in_regime[i]` 且 `c[i] > stop_exit_px` 才准重進；另加 `reentry_min_bars: int = 0`（停損後至少隔 N 根）。
- 語意：`armed` 從布林改成三態（armed / wait_reset / wait_price），`stop_exit_px` 在趨勢重置時清空。
- 動機：exit_ema50_chandelier_immediate 顯示 low 觸價 + 立即重進 76–78% 的重進價高於停損價（ETH 平均溢價 +3.0%），這是機制缺陷不是參數；不修引擎，任何停損類變體都無法公平測試。
- 預期：讓「停損只截斷真正的崩盤」可被檢驗；第 9 節方向 2 依賴此提案。

### P2. 資金費率與執行延遲納入引擎

- 新欄位：`funding_rate_8h: float = 0.0`（每根日線扣 `exposure[i] × funding_rate_8h × 3 × 權益`，在收盤結算後、算 dd 之前）；進階版接受 `funding: np.ndarray | None` 逐日陣列（之後由 data.py 抓 Binance 歷史費率）。
- 新欄位：`exec_delay_bars: int = 0`（pending 在第 i+1+delay 根開盤成交；用 deque 保存多個 pending，後到的覆蓋前面同方向的）。
- `evaluate_variant.py` 加 `--stress` 開關：一次跑（費 0.10% + 滑 0.05%）、（費 0.10% + 滑 0.20%）、（funding 0.01%/8h）、（delay 1）四組並在報表附一列。
- 動機：兩位可執行審查者指出資料是 USDT-M 永續、策略約 50% 時間做多且集中在牛市（費率最高時段）；funding 0.01%/8h 下冠軍 BTC OOS 掉到 1.17 < 1.18；vol_downside_gate 的 ETH 回撤優勢在延遲 1 根後從 −48% 退回 −64%。這類對成交時點高度敏感的優勢應在收錄前揭露。
- 預期：收錄條件從「回測過閘」升級為「壓力情境仍過閘」，避免上線後被成本吃掉。

### P3. 尺度無關的波動目標（相對 + 下行）

- 新欄位：`vol_target_mode: str = "absolute"`（現行）｜`"relative"`：`target = min(max_exposure, vol_target × ref[i] / rvol[i])`，其中 `ref[i] = median(rvol[i-vol_ref_lookback+1 .. i])`，`vol_ref_lookback: int = 365`；此時 `vol_target` 的語意變成「rvol 高於自身一年中位數幾倍才開始砍」（例如 1.5）。
- 新欄位：`vol_side: str = "total"`｜`"downside"`：rvol 只用負報酬的半變異（`sqrt(mean(min(r,0)^2))`），上行波動不砍倉。
- 動機：vol_shock_budget 的 0.70 是 IS 校準（BTC IS rvol p65 0.71、OOS p65 0.47），BTC 2025–2026 預算 0 天生效；ETH 2021-01（+78% 月）被砍到 0.54 曝險損失 −15.5%；波動同時聚集在崩盤與噴出段，總波動的報酬不對稱不穩定。
- 預期：同一組參數在 BTC/ETH、2020/2026 有相同語意；只對下跌波動反應，讓波動預算不在牛市加速段付成本。第 9 節方向 4 依賴此提案。

### P4. 引擎層狀態機：分離進場 / 出場訊號 + 回撤暫停解除規則

- `simulate_regime` 的 `in_regime` 參數擴充為可接受 `dict(enter=np.ndarray[bool], exit=np.ndarray[bool])`；引擎維護 `state`：`not state and enter[i]` → True，`state and exit[i]` → False，其他維持；回傳的 `extra` 加 `regime_entries`、`regime_exits`。舊的單一布林陣列語意不變（enter = in_regime、exit = ~in_regime）。
- `dd_pause` 加解除規則 `dd_pause_resume: str = "regime_reset"`（現行）｜`"new_high"`（收盤創暫停以來 N 日新高即恢復，`dd_resume_high_n: int = 20`）｜`"bars"`（`dd_resume_bars` 根後恢復）。
- 動機：本輪三個過閘門的變體（channel、downside_gate、band）全部自己寫狀態機，每個都要驗證者重做截斷測試；引擎統一實作後只驗一次，變體檔只剩指標。回撤暫停現行「等趨勢重置」與 ema50_atr 的 reset 同樣是「錯過續漲」結構，ema50_voltarget 1/6 有一半來自這裡。
- 預期：降低變體實作出錯面；讓「回撤暫停」可以在下一輪被公平測試。

## 9. 下一輪值得試的方向

避開已證明無效的：時間停滯出場、low 觸價停損 + 立即重進、月頻判定（TSMOM / Faber）、均線對均線交叉、家族多數決 k=3、Donchian 多尺度多數決、EMA 階梯投票、絕對波動目標短窗。

1. **冠軍 + 崩盤閘（帶絕對波動下限）**：`in_regime = channel_state & ~crash`，crash 沿用 downside_gate 的 20 日 dn/up 半變異比值，但加 `dvol_annualized > 自身 250 日中位數 × 1.25` 的絕對門檻，避免平靜市場的邊緣觸發（BTC 2021-01、2023-09 那種）。假設：ETH 的 IS 1.43→1.5+、BTC 不劣化。判準：BTC 六格（兩切分）不得低於冠軍 0.05 以上。
2. **冠軍 + 收盤確認吊燈停損（依賴 P1）**：channel 進出不變，加 `atr_stop_mult=3, stop_trigger="close", reentry="close_above_stop"`。假設：停損只截斷 2021-05、2022-05 這類快速崩盤，不再付下影線溢價；ETH OOS 回撤 −48% 有機會壓到 −40% 以下。
3. **冠軍 + 快速重進（處理 V 轉延遲）**：進場條件改為「close > EMA50 + 1.0 ATR **或**（close > EMA50 且 close 創 20 日新高）」，出場不變。假設：2023-01 BTC、2024-11 這類穿越後不回頭的 V 型行情少付 1 ATR 的延遲成本，OOS 淨報酬追回持有一部分；風險是 2022 熊市反彈裡 20 日新高更常出現，IS 回合數會回升，用 IS 判斷是否值得。
4. **EMA50 + 相對下行波動預算（依賴 P3）**：`vol_target_mode="relative", vol_side="downside", vol_target=1.5, vol_ref_lookback=365, vol_lookback=20`。假設：shock_budget 在 IS 的正貢獻（躲 2020-03、2021-05 跟進賣壓）保留，而 OOS 的負貢獻（砍到牛市噴出段）消失，且 BTC 2025 後不再是死碼。
5. **第三標的 holdout + 壓力測試（依賴 P2）**：抓 SOL/USDT、BNB/USDT 日線（2020-09 起），對冠軍、vol_downside_gate、exit_ema50_band 跑同參數、同六格 + 替換切分 + `--stress`。這不是新變體，是解決「同一 OOS 已被 16 個變體選擇過」的問題；若冠軍在第三標的的風險調整閘門也過，才進入紙上前向。

備選（若前面 5 個有餘裕）：雙軌配置——BTC 用 vol_atr_channel、ETH 用 vol_downside_gate（ETH 六格全優、BTC 全劣的證據很一致），以「組合」為單位評估而不是單標的。

## 10. 對評估器 / 迴圈流程的建議（不改 regime.py）

1. 「替換切分 OOS Sharpe 掉 > 0.3 判 fail」改成「**超額 Sharpe**（策略 − 同期持有，或 − 基準）掉 > 0.3」；現行規則在 2024 年後會連冠軍與基準一起判掉（持有本身 −0.43）。
2. 把替換切分（IS 2020-07→2023-12 / OOS 2024-01→）與 ±50% 擾動納入 `evaluate_variant.py` 標準流程（`--alt-split`、`--sens-factors 0.5,0.75,1.25,1.5`），每個變體自動得到，不靠驗證者臨時跑。
3. 報表加「變體 − 基準」逐格 ΔSharpe、Δ回撤與日報酬差的配對 t 值；收錄條件加「OOS 各格不得低於基準 0.05 以上」與「|t| ≥ 2 或滾動 2 年勝率 ≥ 65% 才可取代冠軍」。本輪 vol_downside_gate（BTC 三格全劣）與 ladder（|t| < 0.5）都只看閘門會被誤判為升級。
4. 敏感度對映射到相同 in_regime 的擾動去重（ladder 的 vote_frac=0.375 與預設相同；channel 的 atr_len=10 對 BTC 完全無差異），否則通過率高估。
5. 報表加每視窗回合數；冠軍 IS/OOS 只有 11–17 回合，Sharpe 差 0.1–0.2 在此樣本量下不可分辨，接受變體時應同時看 FULL 與跨切分的超額 Sharpe。
6. 變體檔的 PARAMS 加約束宣告（例如 downside_gate 的 `ratio_off > ratio_on ≥ 0.8`），敏感度工具跳過退化組合。

## 11. 本輪產出檔案

- 變體：`experiments/variants/{vol_atr_channel, vol_downside_gate, vol_shock_budget, ensemble_ema_ladder_vote, exit_ema50_stall, exit_ema50_band, ensemble_donchian_ladder, classic_tsmom_1_3_12, ensemble_family_vote, classic_faber_10m_sma, classic_golden_cross_50_200, exit_ema50_chandelier_immediate}.py`
- 結果：`results/variants/<id>.json` 與 `.txt`
- 本報告：`experiments/rounds/round_1.md`
- 迴圈狀態：`experiments/rounds/loop_state.json`（由編排端更新：champion → 正式化到 `abgt/strategies/`、CLAUDE.md 第 5 節、ledger 加 12 筆）
