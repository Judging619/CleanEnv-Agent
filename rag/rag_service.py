
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.retrieval import HybridRetrievalService
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from contextvars import ContextVar
import re


_last_context_docs_ctx: ContextVar[list[Document]] = ContextVar("last_context_docs", default=[])


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.hybrid_retriever = HybridRetrievalService(self.vector_store.vector_store)
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        return self.hybrid_retriever.retrieve(query)

    @staticmethod
    def _build_source_line(doc: Document, idx: int) -> str:
        md = doc.metadata or {}
        source_name = md.get("source_name", "未知来源")
        chunk_id = md.get("chunk_id", f"unknown_{idx}")
        retrieval_hits = md.get("retrieval_hits", "unknown")
        rerank_score = md.get("rerank_score", "unknown")
        return (
            f"来源={source_name}; chunk_id={chunk_id}; "
            f"召回通道={retrieval_hits}; 重排分={rerank_score}"
        )

    def rag_summarize(self, query: str) -> str:

        context_docs = self.retriever_docs(query)
        _last_context_docs_ctx.set(context_docs)
        if not context_docs:
            return "基于现有资料无法确认。参考：无"

        context = ""
        for idx, doc in enumerate(context_docs, start=1):
            source_line = self._build_source_line(doc, idx)
            context += (
                f"【参考资料{idx}】\n"
                f"来源信息：{source_line}\n"
                f"参考内容：{doc.page_content}\n\n"
            )

        answer = self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )
        refs = ",".join(str(i) for i in range(1, len(context_docs) + 1))
        return self._ensure_reference_suffix(answer, refs)

    def clear_last_context_docs(self):
        _last_context_docs_ctx.set([])

    def get_last_references(self) -> list[dict]:
        references = []
        for idx, doc in enumerate(_last_context_docs_ctx.get(), start=1):
            md = doc.metadata or {}
            references.append(
                {
                    "index": idx,
                    "source_name": md.get("source_name", "未知来源"),
                    "chunk_id": md.get("chunk_id", "unknown"),
                    "retrieval_hits": md.get("retrieval_hits", "unknown"),
                    "rerank_score": md.get("rerank_score", "unknown"),
                    "snippet": (doc.page_content or "")[:220],
                }
            )
        return references

    @staticmethod
    def _ensure_reference_suffix(answer: str, refs: str) -> str:
        text = (answer or "").strip()
        # 已有“参考：”则不重复追加
        if re.search(r"(^|\n)\s*参考：", text):
            return text
        if not text:
            return f"基于现有资料无法确认。参考：{refs}"
        return f"{text}\n参考：{refs}"


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
