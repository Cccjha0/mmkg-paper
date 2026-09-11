"""Build the C07/C08 manuscript tables and review from verified history records."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_a_safe_correction/test_history_review_v1"
PAPER = ROOT / "paper_a_draft"
REPORT = ROOT / "docs/reports/paper_a_test_history_review_2026-09-12.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tex(value):
    return "".join({"_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#"}.get(c, c) for c in value)


def main():
    audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "history_evidence_checks_passed" and not audit["failures"]
    assert not any(audit["current_confirmatory_status"].values())
    for rel, digest in audit["sources"].items():
        assert sha(ROOT / rel) == digest, rel
    timeline = json.loads((OUT / "timeline.json").read_text(encoding="utf-8"))
    objects = json.loads((OUT / "frozen_objects.json").read_text(encoding="utf-8"))
    claims = json.loads((OUT / "claim_disposition.json").read_text(encoding="utf-8"))
    tables = {}

    def table(name, caption, label, width, header, rows):
        path = PAPER / "tables/history" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        body = (r"\begin{table}[tbp]" + "\n" + r"\centering\footnotesize" + "\n" +
                r"\caption{" + caption + "}\n" + r"\label{" + label + "}\n" +
                r"\begin{tabularx}{\linewidth}{@{}p{" + width + r"\linewidth}X@{}}" + "\n" +
                r"\toprule" + "\n" + header + r"\\" + "\n" + r"\midrule" + "\n" +
                "\n".join(a + " & " + b + r"\\[3pt]" for a, b in rows) + "\n" +
                r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n" + r"\end{table}" + "\n")
        path.write_text(body, encoding="utf-8", newline="\n")
        tables[path.relative_to(PAPER).as_posix()] = sha(path)

    for index, subset in enumerate((timeline[:7], timeline[7:]), 1):
        rows = [(r"\texttt{" + e["id"] + "}: " + e["author_date"][5:10] + r"\newline\texttt{" +
                 e["revision"][:7] + "}", r"\textbf{" + tex(e["stage"]) + ".} " + tex(e["summary"]))
                for e in subset]
        table(f"timeline_{index}.tex",
              "Recorded development and TEST history, part " + str(index) +
              ". Dates are 2026 repository author dates (UTC+8), not authenticated run or first-view times. "
              "The supplement records full commits, committer dates, source paths and SHA-256 snapshots.",
              f"tab:history-{index}", ".15", "Record / commit & Evidence and limit", rows)
    rows = [(tex(r["object"]) + r"\newline " + ", ".join(r["evidence"]),
             tex(r["supported"]) + r" \emph{Limit:} " + tex(r["limit"])) for r in objects]
    table("frozen_objects.tex", "Frozen-object inventory and the scope of each claim. Record IDs refer to "
          "Tables~\\ref{tab:history-1}--\\ref{tab:history-2}. Matching a policy to a TEST receipt "
          "does not establish that its design preceded the first inspection of that TEST split.",
          "tab:history-objects", ".23", "Object / records & Supported boundary and limitation", rows)

    text = """# C07/C08：历史 TEST 使用与确认性地位核查（2026-09-12）

本轮最低关闭条件通过“补证据并降调”满足。MKG-W 改为主要后验评估；DB15K 改为次要外部评估（后验），不再声称两者是独立确认性 holdout。这里“外部”只指第二套数据，不代表独立于整个研究开发过程。没有运行训练、重新选参或重评 TEST。

核查源版本：`da34b6632b810dea884d37db0339c4055c9b2be6`。历史文档、锁和结果保持原样；本报告及当前 PDF 取代它们的 confirmatory / secondary replication 证据等级解释，不回写或伪造旧冻结记录。

## 已查清与仍不可查清的事

- 初版 ADC 的 β 上限为 0.20，P3 扩展为 0.50。13 特征、OOF、P3 和锁定代码可按 Git 版本追溯。
- 9 月 3 日是本次定位到的四主 pair ADC TEST 数值及其锁摘要首次入库记录。DEV 锁和 TEST 摘要在同一提交的 source manifest 中出现，不能据此构造一个更早的独立冻结时间。
- 同日历史报告明确承认 DB15K TEST 已用于更早的 score-ensemble 实验。9 月 4 日 AACPI 协议进一步明确两套数据的 TEST 结果都已查看；六 pair 暴露清单也包括当时尚未有 pair-specific TEST 输出的 NativE+AdaMF。该协议的禁止条款针对后续 AACPI，不能倒推它证明了早期 ADC 首次查看的精确时间。
- 当前修复特征边界后的结果、健康 DynaSemble、半径分析和 matched alternatives 均晚于已记录的历史曝光。9 月 9 日 DEV-only primary 审计无法追溯性地预注册 9 月 3 日已报告的四 pair。
- 4 个旧策略的锁与 TEST receipt 对照当时已提交的 SHA-256 完全一致，10 个旧/新策略的完整 policy 和 model hash 均匹配。锁定执行身份有证据；方法、feature family、β 上限与 pair frame 在首次查看 TEST 之前已完整冻结，没有足够证据。
- 两套数据首次被人查看的准确时间，以及旧 TEST 是否实际影响方法/特征/pair 选择，均不能从现有日志确定。历史报告写过“没有影响”，但缺少完整、不可变的设计决策/访问记录来独立支持此断言。因此不把“不能排除”写成“已证明发生 TEST 调参”，也不再保证“完全没有影响”。

这里分开三种边界：训练和显式参数选择用哪些标签；filtered evaluation 用哪些已知事实；研究人员是否已见 TEST 结果。默认历史 DEV evaluator 的 TRAIN∪DEV∪TEST filter 是已知事实访问，不能写成“DEV 完全不打开 TEST”；它与 TEST MRR 是否影响设计是不同问题。当前推理特征不接收 filter/gold 的证据由 B07/C04 审计负责。

77 项历史检查通过只说明其检查的执行条件，并不能恢复 holdout 独立性；后续 B07/C04 修复也表明不能把那个 PASS 当作总体无泄漏证明。现有 bootstrap 为固定模型/设计的条件性、逐项描述区间，不覆盖历史 TEST 复用和自适应设计带来的不确定性。

## 可追溯时间线

以下日期均是 Git 作者/提交元数据，不是经认证的运行时间。文件 mtime、目录名日期和脚本“TEST_NOT_USED”自述不被提升为首次查看记录。完整原始证据快照随 `test_history_review_v1/evidence/` 提交。

"""
    for event in timeline:
        text += f"### {event['id']} · {event['stage']}\n\n"
        text += f"作者时间 `{event['author_date']}`；提交时间 `{event['committer_date']}`。\n\n"
        text += event["summary"] + "\n\n"
        for evidence in event["evidence"]:
            url = f"https://github.com/Cccjha0/mmkg-paper/blob/{event['revision']}/{evidence['repository_path']}"
            text += f"- [{evidence['repository_path']}]({url})；SHA-256 `{evidence['sha256']}`。\n"
        text += "\n"
    text += "## 冻结对象与证据范围\n\n"
    for record in objects:
        text += f"- **{record['object']}** ({', '.join(record['evidence'])}): {record['supported']} Limit: {record['limit']}\n"
    text += "\n## 主张处理\n\n"
    for claim in claims:
        text += f"- **{claim['disposition']}** — {claim['claim']} → {claim['replacement']}\n"
    text += """
## 复核与下一次真正确认性实验

轻量核查命令（PowerShell；不训练模型）：

```powershell
python scripts/audit_paper_a_test_history.py
python scripts/build_paper_a_test_history_assets.py
python -m pytest tests/test_test_history_audit.py -q
python paper_a_draft/scripts/compile.py
python paper_a_draft/scripts/verify_draft.py
```

`audit.json` 记录 15 个历史节点、4 个旧策略和 6 个当前策略的核验、未知首次查看时间及明确降调；`policy_bindings.json` 保留每个完整锁、模型、TEST receipt 的 SHA-256。既有大规模源资产的完整检查仍由 draft verifier 负责。

本轮不需要服务器计算。再次运行同一 TEST 不会恢复确认性。若将来需要独立确认，应先将完整方法/feature contract、β/τ/α 搜索空间、候选 pair 和 primary 规则、特征来源/模型 checkpoint 规则、OOF、预算、终点和报告规则一并冻结并留存可追溯提交，再访问一套对该设计仍未暴露的评估集；当前记录不为任何候选数据集预先认证“未暴露”。
"""
    REPORT.write_text(text, encoding="utf-8", newline="\n")
    sources = dict(audit["sources"])
    for path in (OUT / "audit.json", REPORT, Path(__file__)):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    (PAPER / "history_source_manifest.json").write_text(json.dumps({
        "version": "test_history_review_v1", "sources": sources, "tables": tables}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(f"Built {len(tables)} history tables and review; bound {len(sources)} sources")


if __name__ == "__main__":
    main()
