"""把 dashboard/live.json 寫進 Google Drive 電腦版的同步資料夾（我的雲端硬碟/ABGT/abgt_live.json）。

claude.ai 上的 dashboard 用使用者自己的 Google Drive connector 讀這個檔（search_files → download_file_content），
所以每日排程不需要碰 artifact 本身。檔案是 ASCII JSON（中文以 \\uXXXX 跳脫），內容只有測試網子帳與紙上帳本，沒有金鑰。

找資料夾的順序：環境變數 ABGT_DRIVE_DIR → 各磁碟機的「My Drive」/「我的雲端硬碟」下的 ABGT。
找不到（Drive 電腦版沒在跑、沒登入，或捷徑不見）就印一行說明並回傳 3：排程照常往下跑，但 notify 會發通知。
帳號：Drive 電腦版登入 tmes.104010；頁面的 connector 是 jasper1689 —— ABGT 資料夾屬於 jasper、共用給 tmes（編輯者），
tmes 在「我的雲端硬碟」加捷徑後，這裡寫的就是 jasper 那份檔案（同一個 file id）。
覆寫既有檔案（不先刪再建），雲端檔案 id 不變，頁面用檔名搜尋也找得到。
"""
from __future__ import annotations

import json
import os
import string
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "dashboard", "live.json")
NAME = "abgt_live.json"
FOLDER_ID = "1m_g-d2SXMlbhlfAUsdHIFpzjOYixziYs"     # jasper1689 的「ABGT」資料夾（共用給 tmes.104010 編輯）
MY_DRIVE_NAMES = ("My Drive", "我的雲端硬碟", "我的云端硬盘")


def find_dir() -> str | None:
    env = os.environ.get("ABGT_DRIVE_DIR")
    if env and os.path.isdir(env):
        return env
    for letter in string.ascii_uppercase:
        # Drive 電腦版把「共用資料夾的捷徑」掛成 ABGT.lnk，實體在 <磁碟>:\.shortcut-targets-by-id\<資料夾 id>\ABGT
        d = os.path.join(f"{letter}:" + os.sep, ".shortcut-targets-by-id", FOLDER_ID, "ABGT")
        if os.path.isdir(d):
            return d
        for name in MY_DRIVE_NAMES:
            base = f"{letter}:\\{name}"
            d = os.path.join(base, "ABGT")
            if os.path.isdir(d):                 # 不自動建立：ABGT 必須是 jasper 帳號共用過來的資料夾捷徑，否則頁面讀不到
                return d
    return None


def main() -> int:
    if not os.path.exists(SRC):
        print(f"drive_sync：找不到 {SRC}，先跑 build_dashboard.py")
        return 0
    d = find_dir()
    if d is None:
        print("drive_sync：找不到「我的雲端硬碟/ABGT」（Drive 電腦版沒在跑 / 未登入，或共用資料夾捷徑不見了），略過")
        return 3                                                   # 非零 → run_daily.bat 記為失敗步驟、notify 會通知
    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)
    text = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    dst = os.path.join(d, NAME)
    with open(dst, "w", encoding="ascii", newline="") as f:      # 原地覆寫，保留雲端檔案 id
        f.write(text)
    print(f"drive_sync：{dst}（{len(text)} bytes，updated_at {data.get('updated_at')}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
