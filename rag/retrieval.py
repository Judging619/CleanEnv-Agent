import re
import os
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config_handler import chroma_conf
from utils.file_handler import listdir_with_allowed_type, pdf_loader, txt_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    lowered = text.lower()
    latin_tokens = re.findall(r"[a-z0-9_]+", lowered)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]", lowered)
    return latin_tokens + cjk_tokens


@dataclass
class _ScoredDoc:
    doc: Document
    score: float


class HybridRetrievalService:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.vector_k = int(chroma_conf.get("vector_k", 6))
        self.keyword_k = int(chroma_conf.get("keyword_k", 6))
        self.final_k = int(chroma_conf.get("k", 3))
        self.vector_weight = float(chroma_conf.get("vector_weight", 0.65))
        self.keyword_weight = float(chroma_conf.get("keyword_weight", 0.35))
        self._local_chunks_cache: list[Document] | None = None
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )
        self.query_expand_map = {
            "回充": "充电座 返回基站",
            "漏扫": "遗漏 补扫 边角",
            "不出水": "水箱 出水管 拖地",
            "异响": "噪音 卡顿 缠绕",
            "建图": "地图 错乱 重建",
            "续航": "电池 充电 衰减",
        }

    @staticmethod
    def _doc_key(doc: Document) -> str:
        md = doc.metadata or {}
        if md.get("chunk_id"):
            return str(md["chunk_id"])
        source = str(md.get("source_name", ""))
        return f"{source}:{hash(doc.page_content)}"

    def _keyword_score(self, query: str, page_content: str) -> float:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return 0.0

        content = (page_content or "").lower()
        if not content:
            return 0.0

        overlap = sum(1 for token in query_tokens if token in content)
        phrase_bonus = 2.0 if query.strip() and query.strip().lower() in content else 0.0
        return (overlap / len(query_tokens)) + phrase_bonus

    def _rewrite_query_candidates(self, query: str) -> list[str]:
        query = (query or "").strip()
        if not query:
            return []

        candidates = [query]
        for key, expansion in self.query_expand_map.items():
            if key in query:
                candidates.append(f"{query} {expansion}")

        dedup = []
        seen = set()
        for c in candidates:
            norm = c.lower().strip()
            if norm and norm not in seen:
                dedup.append(c)
                seen.add(norm)
        return dedup

    def _vector_retrieve(self, query: str) -> list[Document]:
        query_candidates = self._rewrite_query_candidates(query)
        merged = {}
        try:
            for q in query_candidates:
                docs = self.vector_store.similarity_search(q, k=self.vector_k)
                for doc in docs:
                    merged[self._doc_key(doc)] = doc
                if len(merged) >= self.vector_k:
                    break
            return list(merged.values())[: self.vector_k]
        except Exception as e:
            logger.warning(f"[hybrid_retrieval]向量检索失败，已降级为关键词检索：{str(e)}")
            return []

    def _load_local_chunks(self) -> list[Document]:
        if self._local_chunks_cache is not None:
            return self._local_chunks_cache

        data_root = get_abs_path(chroma_conf["data_path"])
        paths = listdir_with_allowed_type(data_root, tuple(chroma_conf["allow_knowledge_file_type"]))
        chunks: list[Document] = []
        for path in paths:
            try:
                if path.endswith("txt"):
                    docs = txt_loader(path)
                elif path.endswith("pdf"):
                    docs = pdf_loader(path)
                else:
                    docs = []
                split_docs = self._splitter.split_documents(docs)
                for idx, doc in enumerate(split_docs):
                    md = dict(doc.metadata or {})
                    md.setdefault("source_file", path)
                    md.setdefault("source_name", os.path.basename(path))
                    md.setdefault("chunk_id", f"local_{hash(path)}_{idx}")
                    doc.metadata = md
                chunks.extend(split_docs)
            except Exception as e:
                logger.error(f"[hybrid_retrieval]本地关键词库加载失败：{path} | {str(e)}")

        self._local_chunks_cache = chunks
        return chunks

    def _keyword_retrieve(self, query: str) -> list[Document]:
        query_candidates = self._rewrite_query_candidates(query)
        merged_query = " ".join(query_candidates) if query_candidates else query
        source_docs: list[Document] = []
        try:
            raw = self.vector_store.get(include=["documents", "metadatas"])
            documents = raw.get("documents", []) or []
            metadatas = raw.get("metadatas", []) or []
            for idx, page_content in enumerate(documents):
                metadata = metadatas[idx] if idx < len(metadatas) else {}
                source_docs.append(Document(page_content=page_content, metadata=metadata or {}))
        except Exception as e:
            logger.warning(f"[hybrid_retrieval]读取向量库存量数据失败，改用本地知识文件：{str(e)}")
            source_docs = []

        if not source_docs:
            source_docs = self._load_local_chunks()

        candidates: list[_ScoredDoc] = []
        for doc in source_docs:
            page_content = doc.page_content
            score = self._keyword_score(merged_query, page_content)
            if score <= 0:
                continue
            candidates.append(
                _ScoredDoc(doc=Document(page_content=page_content, metadata=dict(doc.metadata or {})), score=score)
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return [item.doc for item in candidates[: self.keyword_k]]

    def _rerank(self, query: str, vector_docs: list[Document], keyword_docs: list[Document]) -> list[Document]:
        merged: dict[str, dict] = {}

        def add_doc(doc: Document, rank: int, limit: int, weight: float, channel: str):
            key = self._doc_key(doc)
            base = merged.setdefault(
                key,
                {
                    "doc": Document(page_content=doc.page_content, metadata=dict(doc.metadata or {})),
                    "score": 0.0,
                    "hits": set(),
                },
            )
            rank_score = (max(limit - rank, 0) / max(limit, 1)) * weight
            base["score"] += rank_score
            base["hits"].add(channel)

        for idx, doc in enumerate(vector_docs):
            add_doc(doc, idx, self.vector_k, self.vector_weight, "vector")

        for idx, doc in enumerate(keyword_docs):
            add_doc(doc, idx, self.keyword_k, self.keyword_weight, "keyword")

        ranked: list[_ScoredDoc] = []
        for item in merged.values():
            extra_score = self._keyword_score(query, item["doc"].page_content) * 0.15
            final_score = item["score"] + extra_score
            md = dict(item["doc"].metadata or {})
            md["retrieval_hits"] = ",".join(sorted(item["hits"]))
            md["rerank_score"] = round(final_score, 4)
            item["doc"].metadata = md
            ranked.append(_ScoredDoc(doc=item["doc"], score=final_score))

        ranked.sort(key=lambda x: x.score, reverse=True)
        return [item.doc for item in ranked[: self.final_k]]

    def retrieve(self, query: str) -> list[Document]:
        vector_docs = self._vector_retrieve(query)
        keyword_docs = self._keyword_retrieve(query)
        return self._rerank(query, vector_docs, keyword_docs)
