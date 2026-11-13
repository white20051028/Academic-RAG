"""Agent 兜底：判断层说「资料不够」（C 类）时，让模型自己多查几次。

## 它到底 Agent 在哪

不是「用了 Agent 框架」，是**流程由判断结果驱动，不由代码写死**：

```
代码写死的管道：
  search → judge → 生成                ← 不管中间结果如何，都走这三步

Agent 兜底：
  search → judge（C）
         → 模型决定：读全文还是换词再搜
         → 执行 → judge 再判
         → 够了就生成，不够就再试
         → 3 步还不行就认输，退回指路
```

**模型在决定「下一步做什么」**，这是 Agent 的最小内核。

## 为什么只做两个工具

只保留 `search` 和 `read_full`：

- `ask_user` 会让体验倒退（现在是「一次列完规则 + 一句引导」，改成
  「停下来问你三轮」，用户要多等），而且 Streamlit 里实现「暂停等用户」
  要管一堆 session state。
- `calc` 收益为零：模型算「155+15」「3.0-2.8」从来不会错；真正难的
  「按培养方案算该修多少学分」是检索问题，不是算术问题。

## 为什么第 1 步固定 read_full

用户反馈的原始案例：问「大三绩点4.0能转专业吗」，答「能」。

根因不是没查到——top-10 里 4 条都带年级限制。是**chunk 割裂**：
「绩点3.00」在第二条，「限定本科一年级」在第四条，模型看到的是碎片。

**再 search 一次拿到的还是碎片。** 只有读完整文件，条款之间的关系才出来。

所以第 1 步不给模型选，直接 read_full；第 2 步才让它选。
"""

from pathlib import Path

MAX_STEPS = 3          # 步数上限。1 步等于现在的 RAG，3 步够覆盖典型路径
MAX_FULL_CHARS = 6000  # 单份文件最多读这么多字，防止撑爆上下文


def find_doc_path(source: str, processed_dir: Path) -> Path | None:
    """按 chunks 里的 source 找到 processed 下的原文件。

    source 是文件名（含 .txt），processed_dir 里是同名文件。
    找不到返回 None——调用方要能处理「文件不存在」这种情况。
    """
    if not source:
        return None
    name = source if source.endswith(".txt") else f"{source}.txt"
    p = processed_dir / name
    return p if p.exists() else None


def read_full(source: str, processed_dir: Path, max_chars: int = MAX_FULL_CHARS) -> str:
    """工具 1：读一份完整文件。

    为什么需要它：chunk 是切开的，模型的上下文里只有碎片。
    「限定本科一年级」和「绩点不低于3.00」在原文里是前后两条，
    但在 chunk 库里是两个不相干的片段。

    读不到就返回空串，不抛异常——Agent 循环里一次工具失败不该终止整个流程。
    """
    p = find_doc_path(source, processed_dir)
    if not p:
        return ""
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return ""
    if len(text) > max_chars:
        # 截断而不是报错。开头通常是标题和总则，比后半段更重要
        text = text[:max_chars] + f"\n\n……（原文共 {len(text)} 字，这里只取前 {max_chars} 字）"
    return text


def pick_next_action(question: str, step: int, tried_sources: list[str],
                     results: list[dict], client, model: str) -> dict:
    """工具 2 的选择器：让模型决定下一步做什么。

    第 1 步不给它选（固定 read_full，理由见模块 docstring）。
    第 2 步开始让它选，只给两个动作：
      READ  <文件名>   —— 换一份文件读全文
      SEARCH <查询词>  —— 换一组词再检索

    为什么不用 JSON：模型输出 JSON 的格式错误率不低（括号、引号、多字段），
    每错一次就要写兜底。单行指令「READ xxx」解析简单、错了也好回退。
    """
    if step == 1:
        # 固定动作：读最相关那份文件的全文
        src = results[0]["meta"]["source"] if results else ""
        return {"action": "read", "arg": src, "why": "第一步固定读完整文件（chunk 割裂）"}

    # 第 2 步起交给模型
    seen = "、".join(tried_sources[-3:]) or "（无）"
    prompt = (
        "你在帮用户查西安科技大学的教务规定。已经查过一轮，但资料还不够。\n\n"
        f"用户问题：{question}\n\n"
        f"已经试过的文件：{seen}\n\n"
        "你可以做两件事之一：\n"
        "  READ   换一份文件读全文（比如读《某某管理办法》）\n"
        "  SEARCH 换一组关键词再检索一次\n\n"
        "只输出一行，格式严格是：\n"
        "READ 文件名.txt\n"
        "或\n"
        "SEARCH 新的一组关键词\n"
    )
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0,
        )
        line = (r.choices[0].message.content or "").strip().split("\n")[0].strip()
    except Exception:  # noqa: BLE001
        line = ""

    if line.upper().startswith("READ"):
        return {"action": "read", "arg": line[4:].strip(), "why": "模型选择读全文"}
    if line.upper().startswith("SEARCH"):
        return {"action": "search", "arg": line[6:].strip(), "why": "模型选择换词再搜"}
    # 模型没按格式输出 → 保守退回「换词再搜」，比读错文件安全
    return {"action": "search", "arg": question, "why": "模型输出无法解析，退回重搜"}


def agent_loop(question: str, results: list[dict], context: str,
               processed_dir: Path,
               judge_classify, judge_client, judge_model,
               search_fn) -> dict:
    """C 类的兜底循环。返回 {"ok", "context", "steps", "note"}。

    每步都重新判断一次（judge_classify）——**这是流程被判断结果驱动的关键**。
    不是「跑完三步再看结果」，是「每步之后都问一次够不够」。

    3 步后仍不够就认输（ok=False），调用方负责退回指路——
    **Agent 只是多给几次机会，不保证能答**。
    """
    steps = []
    tried_sources = []
    tried_queries = []
    cur_context = context

    for step in range(1, MAX_STEPS + 1):
        action = pick_next_action(question, step, tried_sources, results,
                                  judge_client, judge_model)
        act, arg = action["action"], action["arg"]

        if act == "read":
            if not arg or arg in tried_sources:
                # 同一份文件别读两遍：要么换一份，要么改搜
                steps.append({"step": step, "action": "read", "arg": arg, "result": "跳过（已读过）"})
                continue
            text = read_full(arg, processed_dir)
            if not text:
                steps.append({"step": step, "action": "read", "arg": arg, "result": "文件不存在"})
                continue
            tried_sources.append(arg)
            cur_context = f"{cur_context}\n\n【完整文件：{arg}】\n{text}"
            steps.append({"step": step, "action": "read", "arg": arg,
                          "result": f"读到 {len(text)} 字"})
        else:  # search
            # **防重复查询**：实测模型会给两次一样的查询词（「通识教育 学分 培养方案 总学分」），
            # 白烧一步。所以和上次一样时，强制换个角度——加「合计/汇总」这类词，
            # 因为「缺总数」的问题往往要的是汇总口径。
            if arg.strip() and tried_queries and arg.strip() == tried_queries[-1].strip():
                arg = f"{arg} 合计 汇总"
            tried_queries.append(arg)
            try:
                new_results, _ = search_fn(arg)
            except Exception:  # noqa: BLE001
                steps.append({"step": step, "action": "search", "arg": arg, "result": "检索失败"})
                continue
            if new_results:
                results = new_results
                add = "\n\n".join(
                    f"[补{i}] 《{x['meta']['source']}》\n{x['doc']}"
                    for i, x in enumerate(new_results, 1)
                )
                cur_context = f"{cur_context}\n\n{add}"
            steps.append({"step": step, "action": "search", "arg": arg,
                          "result": f"新增 {len(new_results)} 条"})

        # 每步之后重新判断
        kind, reason = judge_classify(question, cur_context, judge_client, judge_model)
        steps[-1]["judge"] = f"{kind}：{reason[:50]}"
        if kind == "B":
            return {"ok": True, "context": cur_context, "steps": steps,
                    "note": f"第 {step} 步找到答案：{reason[:60]}"}

    return {"ok": False, "context": cur_context, "steps": steps,
            "note": f"{MAX_STEPS} 步之后资料仍然不够，退回指路"}
