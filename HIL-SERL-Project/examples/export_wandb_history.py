# 你现在应该怎么确认最新 W&B run 是哪个？

# 在服务器上运行这个脚本，列出最近的 run：

# python - <<'PY'
# import wandb

# api = wandb.Api()
# runs = api.runs("erenjaeger-hit/hil-serl", per_page=20)

# for r in runs:
#     name = r.name
#     if "galaxea_usb_insertion_single" not in name:
#         continue

#     step = r.summary.get("_step", None)
#     print("=" * 80)
#     print("id       :", r.id)
#     print("name     :", r.name)
#     print("state    :", r.state)
#     print("created  :", r.created_at)
#     print("url      :", r.url)
#     print("summary _step:", step)
# PY

# 你要找的是：

# summary _step 接近 6000 或更大
# state 可能是 running / finished / killed
# created 时间更晚

# 找到新的 run id 后，把 export_wandb_history.py 里的：

# RUN_PATH = "erenjaeger-hit/hil-serl/galaxea_usb_insertion_single_20260426_144407"

# 改成新的 run：

# RUN_PATH = "erenjaeger-hit/hil-serl/新的_run_id"

# 然后重新导出。

import csv
import wandb

#中断后，使用上面的指令查询新的地址，分析最新的输出
RUN_PATH = "erenjaeger-hit/hil-serl/galaxea_usb_insertion_single_20260426_154227"
OUT_CSV = "wandb_history_galaxea_usb_insertion_single_20260426_154227.csv"

api = wandb.Api(timeout=60)
run = api.run(RUN_PATH)

print("Run name:", run.name)
print("Run state:", run.state)
print("Run URL:", run.url)

rows = list(run.scan_history(page_size=1000))
print("history rows:", len(rows))

if rows:
    keys = sorted(set().union(*(row.keys() for row in rows)))
    print("columns:")
    for k in keys:
        print(" ", k)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("saved:", OUT_CSV)
else:
    print("没有 history 数据。")