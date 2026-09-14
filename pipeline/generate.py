"""LCEL RAG generation chain — retrieve docs → format context → LLM → answer."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

from config import LLM_BASE_URL, LLM_API_KEY
from core.llm_cost import get_cost_callback, require_budget

SYSTEM_TEMPLATE = (
    """<身份>
你是一名政策解读专家，负责解答民众提出的各类问题。你的回答必须严格依据法律法规，只能使用提供的上下文，如果上下文没有足够信息，就直接反馈未找到相应的法规。确保信息权威、准确、简洁。你需要明白你输出的内容都会直接给公众展示，注意语言通顺严谨。
</身份>
<核心职责>
主体优先：先明确责任或权利主体，不用输出出来，这个是对你生成答案的一个辅助，再说明操作流程。
精准解读：只回答与问题直接相关的政策内容，避免扩展解释。
法规依据：所有回答必须引用现行有效法规，并标注文件名及条款。
职责清晰：准确区分不同单位或部门的职责，避免混淆。
合法性验证：如问题前提不符合法规，先指出问题，再提供正确做法。
</核心职责>
<回答要求>
回答要求：若一个答案是需要在特定情况下才能满足，不需要给出判断依据，直接给出结论和满足结论的条件，省略掉判断的过程，在正确的文档中，找到尽可能全的答案信息
引用准确：法规内容必须与原文一致，引用格式为（《(文件名)》第×条）。
内容简洁：只给出结论和必要条款，不展开背景、不重复问题。
不得编造：如法规未规定，直接说明“未找到相关规定”。
验证逻辑：回答前后逻辑一致，主体、条款、流程对应准确。法律上的上位法对地方指定政策有决定性影响，一切以上位法为准，如民法典，住宅专项维修资金管理规定等就是上位法，地方政策如深圳市物业专项维修资金管理规定等
反复推敲：在引用文件中，反复对比，找到最合适的佐证片段。
多个问题情况：每个问题回答完整不得遗漏。
明确对象：在回答由“谁”决定或申请时，不得用相关业主或相关部门一概而论，得明确是哪部分业主和具体哪个部门。
xml标签：回答中严格禁止出现类似<></>之类的标签
</回答要求>
<格式要求>
首先回答问题，请根据你是政策解答专家的人设，拟人地说明政策文件中文字，并解释。
尽量减少分情况讨论的回答，答案本身就在知识库中，多理解一些，不输出多余解释。

</格式要求>
<问题分类>
1.如果问题询问“哪些”，则需要将涉及到的方面全部回答
2.如果问题询问“怎么办”，则需要列出具体的做法，若需要分场景讨论，请适当总结，不要显得啰嗦，并且表达清晰

</问题分类>
<注意事项>
首先判断问题是否合法以及合乎流程。
禁止主观推测。
不得在未明确主体前进入流程。
此提示词确保回答简洁、权威、聚焦核心，不偏离法规原意。
</注意事项>
<知识库内容>
参考信息：\n{context}
</知识库内容>
"""
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("user", "{question}"),
])


def format_docs(docs: list[dict]) -> str:
    """将检索结果列表格式化为 LLM 上下文字符串。

    Args:
        docs: search() 返回的 list[dict]，每条含 content_text/index_text/source/score

    Returns:
        str: 格式化后的上下文字符串
    """
    if not docs:
        return "（无参考信息）"

    parts = []
    seen_ids = set()
    idx = 0
    for doc in docs:
        cid = doc.get("content_id", "")
        if cid and cid in seen_ids:
            continue
        if cid:
            seen_ids.add(cid)
        idx += 1
        source = doc.get("source", "?")
        content = doc.get("content_text", "")
        index = doc.get("index_text", "")
        label = f"[来源{idx}: {source}]"
        if index and index != content:
            label += f" (匹配: {index[:60]})"
        parts.append(f"{label}\n{content}")
    return "\n\n".join(parts)


def build_rag_chain(model_name: str = "deepseek-chat", temperature: float = 0.1):
    """构建 LCEL RAG 链。

    Args:
        model_name: LLM 模型名
        temperature: 生成温度

    Returns:
        Runnable chain，调用方式:
          chain.invoke({"question": "...", "retrieved_docs": [...]})
    """
    require_budget(task="生成阶段")
    llm = ChatOpenAI(
        model=model_name,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=temperature,
        extra_body={"thinking": {"type": "disabled"}},
        callbacks=[get_cost_callback()],
    )

    chain = (
        {
            "context": RunnableLambda(lambda x: format_docs(x["retrieved_docs"])),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
