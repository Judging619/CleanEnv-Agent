import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.rag_service import RagSummarizeService
from rag.retrieval import HybridRetrievalService
from rag.vector_store import VectorStoreService


@dataclass
class EvalStats:
    total: int
    hit_count: int
    mrr_sum: float
    empty_count: int
    error_count: int
    latencies_ms: list[float]
    doc_counts: list[int]

    @property
    def hit_rate(self) -> float:
        return (self.hit_count / self.total) if self.total else 0.0

    @property
    def mrr(self) -> float:
        return (self.mrr_sum / self.total) if self.total else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(0.95 * (len(sorted_lat) - 1))
        return sorted_lat[idx]

    @property
    def avg_docs(self) -> float:
        return mean(self.doc_counts) if self.doc_counts else 0.0

    @property
    def empty_rate(self) -> float:
        return (self.empty_count / self.total) if self.total else 0.0

    @property
    def error_rate(self) -> float:
        return (self.error_count / self.total) if self.total else 0.0


@dataclass
class AnswerEvalStats:
    total: int
    success_count: int
    error_count: int
    has_reference_count: int
    latencies_ms: list[float]
    keyword_coverages: list[float]
    keyword_cases: int

    @property
    def success_rate(self) -> float:
        return (self.success_count / self.total) if self.total else 0.0

    @property
    def error_rate(self) -> float:
        return (self.error_count / self.total) if self.total else 0.0

    @property
    def reference_rate(self) -> float:
        return (self.has_reference_count / self.total) if self.total else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def avg_keyword_coverage(self) -> float:
        return mean(self.keyword_coverages) if self.keyword_coverages else 0.0


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"dataset第{line_no}行不是合法JSON: {e}") from e

            if "query" not in item or "expected_sources" not in item:
                raise ValueError(f"dataset第{line_no}行缺少query或expected_sources字段")
            rows.append(item)
    return rows


def source_name(doc) -> str:
    md = doc.metadata or {}
    return str(md.get("source_name") or md.get("source") or "")


def first_match_rank(docs, expected_sources: list[str]) -> int:
    expected = [x.lower() for x in expected_sources]
    for idx, doc in enumerate(docs, start=1):
        src = source_name(doc).lower()
        if any(token in src for token in expected):
            return idx
    return 0


def normalize_text(text: str) -> str:
    return (text or "").lower().replace(" ", "")


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float | None:
    if not expected_keywords:
        return None
    ans = normalize_text(answer)
    if not ans:
        return 0.0
    hit = 0
    for kw in expected_keywords:
        if normalize_text(kw) in ans:
            hit += 1
    return hit / len(expected_keywords)


def evaluate(name: str, dataset: list[dict], retrieve_fn):
    details = []
    stats = EvalStats(
        total=len(dataset),
        hit_count=0,
        mrr_sum=0.0,
        empty_count=0,
        error_count=0,
        latencies_ms=[],
        doc_counts=[],
    )

    for item in dataset:
        query = item["query"]
        expected_sources = item["expected_sources"]
        qid = item.get("id", "")

        start = perf_counter()
        error = ""
        docs = []
        try:
            docs = retrieve_fn(query) or []
        except Exception as e:
            error = str(e)
        latency_ms = (perf_counter() - start) * 1000

        rank = first_match_rank(docs, expected_sources)
        hit = rank > 0

        if hit:
            stats.hit_count += 1
            stats.mrr_sum += 1.0 / rank
        if not docs:
            stats.empty_count += 1
        if error:
            stats.error_count += 1

        stats.latencies_ms.append(latency_ms)
        stats.doc_counts.append(len(docs))

        details.append(
            {
                "id": qid,
                "query": query,
                "expected_sources": expected_sources,
                "hit": hit,
                "first_match_rank": rank,
                "latency_ms": round(latency_ms, 2),
                "error": error,
                "retrieved_sources": [source_name(doc) for doc in docs],
            }
        )

    return {"name": name, "stats": stats, "details": details}


def evaluate_answers(dataset: list[dict], answer_fn):
    details = []
    stats = AnswerEvalStats(
        total=len(dataset),
        success_count=0,
        error_count=0,
        has_reference_count=0,
        latencies_ms=[],
        keyword_coverages=[],
        keyword_cases=0,
    )

    for item in dataset:
        query = item["query"]
        qid = item.get("id", "")
        expected_keywords = item.get("expected_keywords", [])

        start = perf_counter()
        error = ""
        answer = ""
        try:
            answer = answer_fn(query) or ""
        except Exception as e:
            error = str(e)
        latency_ms = (perf_counter() - start) * 1000

        if error:
            stats.error_count += 1
        else:
            stats.success_count += 1

        has_ref = "参考：" in answer
        if has_ref:
            stats.has_reference_count += 1

        coverage = keyword_coverage(answer, expected_keywords)
        if coverage is not None:
            stats.keyword_cases += 1
            stats.keyword_coverages.append(coverage)

        stats.latencies_ms.append(latency_ms)
        details.append(
            {
                "id": qid,
                "query": query,
                "expected_keywords": expected_keywords,
                "keyword_coverage": None if coverage is None else round(coverage, 4),
                "has_reference": has_ref,
                "latency_ms": round(latency_ms, 2),
                "error": error,
                "answer": answer,
            }
        )

    return {"name": "answer_eval", "stats": stats, "details": details}


def to_markdown(dataset_path: Path, dense_res: dict, production_res: dict, answer_res: dict | None) -> str:
    b = dense_res["stats"]
    h = production_res["stats"]

    lines = [
        "# 检索策略评测报告",
        "",
        f"- 评测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据集：`{dataset_path}`",
        f"- 样本数：{b.total}",
        "",
        "## 检索策略指标",
        "",
        "| 方案 | Hit@K | MRR | 平均延迟(ms) | P95延迟(ms) | 平均召回条数 | 空结果率 | 错误率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| 策略A（Dense向量检索） | {b.hit_rate:.2%} | {b.mrr:.4f} | {b.avg_latency_ms:.2f} | {b.p95_latency_ms:.2f} | {b.avg_docs:.2f} | {b.empty_rate:.2%} | {b.error_rate:.2%} |",
        f"| 策略B（生产链路：向量+关键词+重排） | {h.hit_rate:.2%} | {h.mrr:.4f} | {h.avg_latency_ms:.2f} | {h.p95_latency_ms:.2f} | {h.avg_docs:.2f} | {h.empty_rate:.2%} | {h.error_rate:.2%} |",
        "",
    ]

    if answer_res is not None:
        a = answer_res["stats"]
        lines.extend(
            [
                "## 回答质量指标（生产回答链路）",
                "",
                "| 指标 | 数值 |",
                "|---|---:|",
                f"| 回答成功率 | {a.success_rate:.2%} |",
                f"| 回答错误率 | {a.error_rate:.2%} |",
                f"| 参考行合规率（包含“参考：”） | {a.reference_rate:.2%} |",
                f"| 平均关键词覆盖率 | {a.avg_keyword_coverage:.2%} |",
                f"| 回答平均延迟(ms) | {a.avg_latency_ms:.2f} |",
                "",
            ]
        )
        if a.error_rate > 0:
            lines.extend(
                [
                    "- 说明：回答评测依赖大模型在线服务；若网络或鉴权异常，回答成功率会下降。",
                    "",
                ]
            )

    lines.extend(
        [
        "## 失败样例（生产链路）",
        "",
        ]
    )

    fail_rows = [x for x in production_res["details"] if not x["hit"]][:10]
    if not fail_rows:
        lines.append("- 无失败样例。")
    else:
        for row in fail_rows:
            lines.append(f"- {row['id']} | {row['query']} | 召回源：{row['retrieved_sources']}")

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 当前项目已具备可复现的检索与回答评测流程，可用于持续迭代知识库与策略参数。",
            "- 建议继续叠加人工抽样评分或LLM Judge，对“建议是否可执行、是否幻觉”做更细粒度评估。",
        ]
    )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="评估项目检索策略与回答链路效果")
    parser.add_argument(
        "--dataset",
        type=str,
        default="eval/dataset.jsonl",
        help="jsonl评测集路径",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="eval/metrics_report.md",
        help="markdown报告输出路径",
    )
    parser.add_argument(
        "--details",
        type=str,
        default="eval/metrics_details.json",
        help="详细结果输出路径",
    )
    parser.add_argument(
        "--with-answer-eval",
        action="store_true",
        help="启用回答级评测（会调用RAG回答链路）",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    report_path = Path(args.report)
    details_path = Path(args.details)

    dataset = load_dataset(dataset_path)
    vs = VectorStoreService()
    baseline_retriever = vs.get_retriever()
    hybrid_retriever = HybridRetrievalService(vs.vector_store)

    dense_res = evaluate("dense_vector_retrieval", dataset, baseline_retriever.invoke)
    production_res = evaluate("production_hybrid_retrieval", dataset, hybrid_retriever.retrieve)
    answer_res = None
    if args.with_answer_eval:
        # 评测场景下关闭RAG提示词回显，避免终端输出过长。
        import rag.rag_service as rag_service_module

        rag_service_module.print_prompt = lambda prompt: prompt
        rag_service = RagSummarizeService()
        answer_res = evaluate_answers(dataset, rag_service.rag_summarize)

    report = to_markdown(dataset_path, dense_res, production_res, answer_res)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    serializable = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "strategy_a_dense": {
            "name": dense_res["name"],
            "stats": {
                "total": dense_res["stats"].total,
                "hit_rate": dense_res["stats"].hit_rate,
                "mrr": dense_res["stats"].mrr,
                "avg_latency_ms": dense_res["stats"].avg_latency_ms,
                "p95_latency_ms": dense_res["stats"].p95_latency_ms,
                "avg_docs": dense_res["stats"].avg_docs,
                "empty_rate": dense_res["stats"].empty_rate,
                "error_rate": dense_res["stats"].error_rate,
            },
            "details": dense_res["details"],
        },
        "strategy_b_production": {
            "name": production_res["name"],
            "stats": {
                "total": production_res["stats"].total,
                "hit_rate": production_res["stats"].hit_rate,
                "mrr": production_res["stats"].mrr,
                "avg_latency_ms": production_res["stats"].avg_latency_ms,
                "p95_latency_ms": production_res["stats"].p95_latency_ms,
                "avg_docs": production_res["stats"].avg_docs,
                "empty_rate": production_res["stats"].empty_rate,
                "error_rate": production_res["stats"].error_rate,
            },
            "details": production_res["details"],
        },
    }
    if answer_res is not None:
        a = answer_res["stats"]
        serializable["answer_eval"] = {
            "name": answer_res["name"],
            "stats": {
                "total": a.total,
                "success_rate": a.success_rate,
                "error_rate": a.error_rate,
                "reference_rate": a.reference_rate,
                "avg_keyword_coverage": a.avg_keyword_coverage,
                "avg_latency_ms": a.avg_latency_ms,
                "keyword_cases": a.keyword_cases,
            },
            "details": answer_res["details"],
        }
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    print("评测完成")
    print(f"报告：{report_path}")
    print(f"详情：{details_path}")
    print(
        "生产检索链路指标 "
        f"Hit@K={production_res['stats'].hit_rate:.2%}, "
        f"MRR={production_res['stats'].mrr:.4f}, "
        f"AvgLatency={production_res['stats'].avg_latency_ms:.2f}ms"
    )
    if answer_res is not None:
        a = answer_res["stats"]
        print(
            "Answer核心指标 "
            f"Success={a.success_rate:.2%}, "
            f"RefRate={a.reference_rate:.2%}, "
            f"KeywordCoverage={a.avg_keyword_coverage:.2%}"
        )


if __name__ == "__main__":
    main()
