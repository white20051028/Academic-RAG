"""查询改写：把口语问题换成「学校文件里会用的书面说法」。

为什么需要：
  实测 BM25 让书面评测 +2 点，但口语评测 -5.3 点。
  原因是 **BM25 的前提是「你说的词，文档里也有」**——
  学生问「挂科了要重修吗」，文档里写的是「不及格」「补考」，
  字面上一个词都对不上，BM25 只能抓到噪声。

改写的定位：
  不是改得"更好看"，而是改得**更贴近文档措辞**。
  所以 few-shot 例子里给的都是校园场景的口语 → 书面映射。

关键设计：两路用不同的查询
  向量路用**原查询**（保住用户真实意图的语义）
  BM25 路用**改写后查询**（保住字面能对上）
  这样两路各用各的长处，而不是强行统一。
"""

REWRITE_PROMPT = """你的任务：把学生问的校园问题，改写成「在学校教务处文件里检索用的关键词」。

要求：
1. 保留原意，但换成学校文件会用的书面说法
2. 补上同义词（口语 → 书面）
3. 只输出一行关键词，用空格分隔，不要解释，不要标点

示例：
问：挂科了要重修吗
答：不及格 补考 重修 学分 课程考核

问：咋转专业
答：转专业 申请条件 办理流程 时间 绩点

问：毕业证找不着了咋办
答：毕业证书 学位证书 补授 遗失 补办

问：我绩点2.5能不能毕业
答：毕业 学位授予 平均学分绩点 2.30 条件

问：请假最多能请多少天
答：请假 期限 手续 审批

问：{question}
答："""


def rewrite_query(question: str, client, model: str = "Qwen/Qwen2.5-7B-Instruct") -> str:
    """把口语问题改写成关键词。失败时退回原问题。"""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
            max_tokens=60,
            temperature=0,
        )
        out = (r.choices[0].message.content or "").strip()
        # 去掉可能的"答："前缀，取第一行
        out = out.split("\n")[0].replace("答：", "").strip()
        return out or question
    except Exception:  # noqa: BLE001
        return question
