#!/usr/bin/env bash
# 跳水线批量：把一个目录下的所有 *_replay_frames.bin.br 各跑一整轮。
#
#   ./scripts/dive_batch.sh DIR [--steps extract,plot,render,overlay] [--skip PAT]
#   ./scripts/dive_batch.sh ~/Downloads/DATAS/DIVING/downloaded_3m_new
#
# 每个 clip 一节，日志落 outputs/batch_<目录名>.log。单个 clip 失败不中断整批
# （单目 replay 里总有几段人没检出来），最后汇总成功/失败。
#
# --skip 给一个 grep 模式，命中的 clip 跳过。比如 20260803-123529 只有 76/600
# 帧可用、体高 0.97 m，HSMR 明显没跟上，跑它只是浪费时间：
#   ./scripts/dive_batch.sh DIR --skip 123529
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE=./scripts/dive.sh

steps="extract,plot,render,overlay"
skip=""
dir=""
while (($#)); do
  case "$1" in
    --steps) steps="$2"; shift 2 ;;
    --skip) skip="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) dir="$1"; shift ;;
  esac
done
[[ -n "$dir" && -d "$dir" ]] || { echo "用法：$0 DIR [--steps …] [--skip PAT]" >&2; exit 2; }

log="outputs/batch_$(basename "$dir").log"
mkdir -p outputs
: > "$log"

# macOS 自带 bash 3.2，没有 `mapfile`/`readarray`。用 while read 攒一个换行分隔的
# 列表，再靠 IFS 拆——与仓库其他脚本一致的可移植写法。
files=""
while IFS= read -r path; do
  [[ -n "$skip" ]] && printf '%s' "$path" | grep -q "$skip" && continue
  files="${files}${path}"$'\n'
done < <(find "$dir" -name '*_replay_frames.bin.br' | sort)

count=$(printf '%s' "$files" | grep -c . || true)
echo "total clips: $count" | tee -a "$log"
((count)) || { echo "没有 *_replay_frames.bin.br"; exit 1; }

ok=0; bad=""
while IFS= read -r frames <&3; do
  [[ -n "$frames" ]] || continue
  stem="$(basename "$frames")"; stem="${stem%%_replay_frames*}"
  echo "===== ${stem} =====" | tee -a "$log"
  # extract 必须带 --frames；其余步骤认 --dir，否则 latest_dive_dir 会指错目录。
  # 两处细节，缺哪个都会静默出错：
  #   `<&3` + `</dev/null`  循环从 fd 3 读，子进程 stdin 接空。共用 stdin 时
  #                         python 会从 here-string 里读走字节，下一轮 read 拿到
  #                         残缺的路径——实测报「frames 不存在：…/haotian/Downloads」，
  #                         每轮多啃几个字符。
  #   `${stem}` 的花括号     bash 3.2 在非 UTF-8 locale 下把全角括号的字节当作合法
  #                         变量名字符，`$stem（` 会被解析成 `${stem（详见}`，
  #                         配合 set -u 直接报 unbound variable。
  rest="${steps/extract,/}"
  if $HERE extract --frames "$frames" >>"$log" 2>&1 </dev/null \
     && { [[ "$rest" == "$steps" ]] || $HERE "$rest" --dir "outputs/dive_${stem}" \
          >>"$log" 2>&1 </dev/null; }; then
    ok=$((ok + 1)); echo "OK ${stem}" | tee -a "$log"
  else
    bad="${bad} ${stem}"; echo "FAIL ${stem} - 详见 $log" | tee -a "$log"
  fi
done 3<<< "$files"

echo "" | tee -a "$log"
echo "成功 $ok / $count" | tee -a "$log"
[[ -n "$bad" ]] && echo "失败：$bad" | tee -a "$log"
echo "日志 $log"
