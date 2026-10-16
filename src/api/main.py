"""
[M12] HTTP 服务（FastAPI）

把 Agent 包成接口。这一步的意义不只是「能对外提供服务」，还有两条更实际的：

    1. 它逼你把 Agent 的输入输出定义成**明确的数据结构**（Pydantic 模型），
       而不是「传个字符串、返回个对象」。
       定义得清楚，别人才接得上你的东西 —— 这是后端能力最朴素的体现。

    2. 接口里可以直接挂上评测能力。/evaluate 这个端点返回三维度指标 ——
       把一个评测能力做成服务，和做成一个跑完就扔的脚本，是两件不同的事。

三个端点：

    GET  /health     健康检查
    POST /chat       一轮对话，返回回复 + 完整执行轨迹
    POST /evaluate   对评测集跑一轮，返回任务完成率 / 工具准确性 / 满意度代理

启动方式（项目根目录）：
    uvicorn src.api.main:app --reload --port 8000
    然后打开 http://127.0.0.1:8000/docs 看自动生成的接口文档
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.cs_agent import CSAgent
from src.agent.graph_agent import GraphAgent
from src.agent.hybrid import HybridRetriever
from src.agent.tools import ToolRuntime
from src.config import LLM_API_KEY, RECORDS_JSON
from src.llm_client import UsageMeter

app = FastAPI(
    title="AI 客服 Agent",
    version="2.0",
    description="可调用工具、可多轮、带执行轨迹的客服 Agent，附带 Agent 评测能力",
)


# ============================================================
# 一、请求 / 响应模型
# ============================================================


class Message(BaseModel):
    role: str = Field(..., description="user 或 assistant")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户这一轮说的话")
    history: list[Message] = Field(default_factory=list, description="之前的对话，可为空")
    engine: str = Field("graph", description="用哪个编排：graph（LangGraph）或 loop（手写）")
    include_trace: bool = Field(False, description="是否返回完整执行轨迹")


class ToolCallOut(BaseModel):
    name: str
    arguments: dict
    ok: bool
    error: str | None = None


class ChatResponse(BaseModel):
    answer: str
    stopped_reason: str
    rounds: int
    total_ms: int
    tools: list[str]
    tool_calls: list[ToolCallOut]
    guard_blocks: int
    usage: dict
    trace: list[dict[str, Any]] | None = None


class EvalRequest(BaseModel):
    origin: str = Field("all", description="跑哪部分：all / measured / multiturn")
    limit: int | None = Field(None, description="只跑前 N 条，便于快速验证")


# ============================================================
# 二、共享资源（进程内只加载一次）
# ============================================================


class Resources:
    """检索器和判例数据。启动时加载一次，避免每次请求都读盘、重建索引。"""

    def __init__(self) -> None:
        self.records: list[dict] = []
        self.retriever: HybridRetriever | None = None
        self.ready = False
        self.error = ""

    def load(self) -> None:
        try:
            self.records = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
            self.retriever = HybridRetriever(self.records, use_rerank=True)
            self.retriever.prepare()
            self.ready = True
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self.ready = False


R = Resources()


@app.on_event("startup")
def _startup() -> None:
    R.load()


# ============================================================
# 三、端点
# ============================================================


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if R.ready else "degraded",
        "llm_configured": bool(LLM_API_KEY),
        "precedents": len(R.retriever) if R.retriever else 0,
        "vector_enabled": bool(R.retriever and R.retriever.vector_enabled),
        "error": R.error,
    }


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    if not R.ready:
        raise HTTPException(status_code=503, detail=f"资源未就绪：{R.error}")
    if not LLM_API_KEY:
        raise HTTPException(status_code=503, detail="未配置 LLM_API_KEY")

    runtime = ToolRuntime(index=R.retriever, exclude_id=None)
    meter = UsageMeter()

    if req.engine == "loop":
        agent = CSAgent(runtime=runtime, meter=meter)
    else:
        agent = GraphAgent(runtime=runtime, meter=meter)

    trace = agent.run(
        question=req.question,
        case_id="adhoc",
        history=[m.model_dump() for m in req.history] or None,
    )

    calls = [
        ToolCallOut(
            name=t["name"],
            arguments=t.get("arguments") or {},
            ok=bool(t.get("ok")),
            error=t.get("error"),
        )
        for t in trace.tool_calls
    ]

    return ChatResponse(
        answer=trace.final_answer,
        stopped_reason=trace.stopped_reason,
        rounds=trace.rounds,
        total_ms=trace.total_ms,
        tools=trace.tool_names,
        tool_calls=calls,
        guard_blocks=len(trace.guard_events),
        usage=trace.usage,
        trace=[s.to_dict() for s in trace.steps] if req.include_trace else None,
    )


@app.post("/evaluate")
def evaluate_endpoint(req: EvalRequest) -> dict:
    """跑一轮评测，返回三维度指标。

    注意这里返回的是**过程 + 结果**两组指标 ——
    只报「答对了多少」的评测，看不出「靠运气对的」有多少。
    """
    if not R.ready:
        raise HTTPException(status_code=503, detail=f"资源未就绪：{R.error}")

    from src.eval.dataset import build_all
    from src.eval.metrics import build_report

    cases = build_all()
    if req.origin != "all":
        cases = [c for c in cases if c.origin == req.origin]
    if req.limit:
        cases = cases[: req.limit]

    traces = []
    verdicts: dict[str, dict] = {}
    for case in cases:
        rt = ToolRuntime(index=R.retriever, exclude_id=case.numeric_id)
        agent = GraphAgent(runtime=rt)
        tr = agent.run(case.question, case_id=case.case_id, history=case.history or None)
        traces.append(tr.to_dict())
        # 这里只做「跑」的部分；打分要另外调评测 Agent，
        # 因为评测本身也是一次模型调用，不该和跑用例混在一个请求里做
        verdicts[case.case_id] = {"score": 4 if tr.stopped_reason == "answered" else 2}

    report = build_report(cases, traces, verdicts)
    report["note"] = "verdicts 此处由 stopped_reason 粗略映射；正式评测请走 src.eval.runner + judge"
    return report
