import os
import time

import streamlit as st

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import (
    clear_last_rag_references,
    get_last_rag_references,
    reload_rag_service,
)
from rag.vector_store import VectorStoreService
from utils.backend_client import BackendClient, BackendClientError
from utils.logger_handler import logger, request_context
from utils.session_store import (
    create_session,
    delete_session,
    get_session,
    list_sessions,
    save_session_messages,
)


BACKEND_URL = os.getenv("ZST_BACKEND_URL", "http://localhost:8000").strip()
USE_BACKEND_API = bool(BACKEND_URL)
ALLOW_LOCAL_AGENT_FALLBACK = os.getenv("ZST_ALLOW_LOCAL_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}
QUICK_PROMPTS = [
    "扫地机器人回充失败怎么办？",
    "我当前位置适合拖地吗？",
    "生成最近一期使用报告",
    "家里有宠物怎么清扫？",
    "滤网多久需要更换？",
]

if "sidebar_open" not in st.session_state:
    st.session_state["sidebar_open"] = True


st.set_page_config(
    page_title="洁境智顾 Agent",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded" if st.session_state["sidebar_open"] else "collapsed",
)


def apply_page_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600;700;900&family=Source+Sans+3:wght@400;600;700&display=swap');

        :root {
            --bg: #f7f4ec;
            --ink: #27312f;
            --muted: #69746f;
            --card: rgba(255, 255, 255, 0.76);
            --line: rgba(39, 49, 47, 0.12);
            --green: #2f6b4f;
            --green-2: #84b59f;
            --clay: #c98255;
            --sand: #eadcc5;
            --shadow: 0 24px 70px rgba(39, 49, 47, 0.13);
        }

        .stApp {
            color: var(--ink);
            background: #fbfbfb;
            font-family: "Source Sans 3", "PingFang SC", sans-serif;
        }

        .block-container {
            max-width: 1120px;
            padding-top: 5.25rem;
            padding-bottom: 3.2rem;
        }

        [data-testid="stSidebar"] {
            background: #f7f7f7;
            color: #1f2328;
            border-right: 1px solid rgba(31, 35, 40, 0.08);
        }

        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 0;
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.85rem;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            margin-top: 0;
        }

        [data-testid="stSidebar"] * {
            color: #1f2328;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-family: "Noto Serif SC", serif;
            letter-spacing: 0.02em;
        }

        [data-testid="stSidebar"] [data-testid="stButton"] button {
            border-radius: 14px;
            border: 1px solid rgba(31, 35, 40, 0.08);
            background: #ffffff;
            color: #1f2328;
            min-height: 2.75rem;
            font-weight: 700;
            box-shadow: none;
            transition: all 160ms ease;
        }

        [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
            background: #f0f0f0;
            border-color: rgba(31, 35, 40, 0.12);
            transform: translateY(-1px);
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(31, 35, 40, 0.08);
            margin: 1.55rem 0;
        }

        .sidebar-section-title {
            margin: 1.2rem 0 0.55rem;
            color: rgba(248, 242, 232, 0.72);
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.16em;
        }

        .sidebar-brand {
            position: sticky;
            top: 0;
            z-index: 80;
            display: flex;
            align-items: center;
            gap: 0.72rem;
            padding: 0.65rem 0 0.75rem;
            margin: 0 0 0.85rem;
            background: #f7f7f7;
            border-bottom: 1px solid rgba(31, 35, 40, 0.08);
        }

        .sidebar-avatar {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.2rem;
            height: 2.2rem;
            border-radius: 999px;
            color: #ffffff;
            background: linear-gradient(135deg, #2f6b4f, #84b59f);
            font-family: "Noto Serif SC", serif;
            font-weight: 900;
            box-shadow: 0 8px 20px rgba(47, 107, 79, 0.22);
        }

        .sidebar-brand-en {
            color: #7a7f84;
            font-size: 0.62rem;
            font-weight: 800;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-top: 0.15rem;
        }

        .sidebar-brand-title {
            font-family: "Noto Serif SC", serif;
            font-size: 1.1rem;
            line-height: 1.05;
            letter-spacing: -0.02em;
            margin: 0;
            color: #1f2328;
            font-weight: 900;
        }

        [data-testid="stChatMessage"] {
            background: var(--card);
            border: 1px solid rgba(255, 255, 255, 0.78);
            border-radius: 26px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.82rem;
            box-shadow: 0 10px 28px rgba(39, 49, 47, 0.06);
            backdrop-filter: blur(14px);
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            font-size: 1.02rem;
            line-height: 1.85;
        }

        [data-testid="stChatMessage"],
        [data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
            color: var(--ink) !important;
        }

        [data-testid="stChatMessage"] a {
            color: #2f6b4f !important;
            font-weight: 700;
        }

        [data-testid="stBottom"],
        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"] {
            background: #fbfbfb !important;
        }

        [data-testid="stBottomBlockContainer"] {
            padding: 0.9rem 0 1rem;
        }

        [data-testid="stChatInput"] {
            max-width: 920px;
            margin: 0 auto;
            background: transparent !important;
        }

        [data-testid="stChatInput"] div {
            background: transparent !important;
        }

        [data-testid="stChatInput"] > div {
            min-height: 3rem;
            border-radius: 999px !important;
            border: 1px solid rgba(47, 107, 79, 0.2) !important;
            background: #ffffff !important;
            box-shadow: 0 10px 28px rgba(39, 49, 47, 0.08);
        }

        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] [data-testid="stChatInputTextArea"] {
            background: transparent !important;
            color: var(--ink) !important;
            caret-color: var(--green) !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #9aa0a6 !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] button,
        [data-testid="stChatInput"] button svg {
            background: transparent !important;
            color: var(--green) !important;
            fill: var(--green) !important;
        }

        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"],
        button[aria-label="Close sidebar"],
        button[aria-label="Open sidebar"],
        button[title="Close sidebar"],
        button[title="Open sidebar"] {
            display: none !important;
            visibility: hidden !important;
        }

        .st-key-fixed_topbar {
            position: fixed;
            top: 0;
            right: 0;
            height: 4.1rem;
            z-index: 900;
            background: rgba(255, 255, 255, 0.96);
            border-bottom: 1px solid rgba(31, 35, 40, 0.08);
            backdrop-filter: blur(18px);
        }

        .st-key-fixed_topbar [data-testid="stHorizontalBlock"] {
            height: 4.1rem;
            align-items: center;
            padding: 0 1.2rem;
        }

        .st-key-fixed_topbar [data-testid="stButton"] button {
            width: 2.35rem;
            height: 2.35rem;
            min-height: 2.35rem;
            padding: 0;
            border-radius: 999px;
            background: #ffffff;
            color: #1f2328;
            border: 1px solid rgba(31, 35, 40, 0.1);
            box-shadow: 0 8px 22px rgba(31, 35, 40, 0.08);
            font-size: 1.28rem;
            font-weight: 800;
        }

        .topbar-title {
            text-align: center;
            font-weight: 800;
            color: #111111;
            line-height: 1.2;
        }

        .topbar-subtitle {
            margin-top: 0.25rem;
            text-align: center;
            color: #b6b6b6;
            font-size: 0.78rem;
            font-weight: 600;
        }

        [data-testid="stExpander"] {
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.62);
        }

        .empty-state {
            margin: 2rem auto 0;
            padding: 1.5rem 1.65rem;
            max-width: 680px;
            border-radius: 28px;
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(255, 255, 255, 0.76);
            color: var(--muted);
            box-shadow: 0 18px 48px rgba(39, 49, 47, 0.08);
        }

        .empty-state strong {
            color: var(--ink);
        }

        .st-key-welcome_panel {
            min-height: 42vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding-bottom: 3rem;
        }

        .welcome-title {
            margin: 0 0 1.8rem;
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 900;
            color: #111111;
            letter-spacing: -0.04em;
        }

        .st-key-quick_prompts {
            max-width: 920px;
            margin: 0 auto;
        }

        .st-key-quick_prompts [data-testid="stButton"] button {
            padding: 0.78rem 1.05rem;
            border-radius: 16px;
            background: #f1f1f1;
            color: #222222;
            font-weight: 650;
            border: 1px solid rgba(31, 35, 40, 0.06);
            box-shadow: none;
            min-height: 3.25rem;
        }

        .st-key-quick_prompts [data-testid="stButton"] button:hover {
            background: #e9ece8;
            border-color: rgba(47, 107, 79, 0.18);
            color: #1f2328;
        }

        .thinking-indicator {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            margin: 1.1rem 0 1.35rem;
            color: #27312f;
            font-size: 1.08rem;
            font-weight: 800;
        }

        .thinking-mark {
            width: 1rem;
            height: 1rem;
            border-radius: 999px;
            background: #2f6b4f;
            box-shadow: 0 0 0 6px rgba(47, 107, 79, 0.12);
            flex: 0 0 auto;
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        @media (max-width: 720px) {
            .block-container {
                padding-top: 1.2rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_page_style()


def apply_dynamic_layout_style():
    topbar_left = "21rem" if st.session_state.get("sidebar_open", True) else "0"
    st.markdown(
        f"""
        <style>
        .st-key-fixed_topbar {{
            left: {topbar_left};
            width: calc(100vw - {topbar_left});
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_dynamic_layout_style()


def use_backend_api() -> bool:
    return USE_BACKEND_API and st.session_state.get("backend_mode", True)


def restore_backend_mode_if_available():
    if not USE_BACKEND_API or st.session_state.get("backend_mode", True):
        return

    try:
        get_backend_client().health()
    except BackendClientError:
        return

    st.session_state["backend_mode"] = True
    st.session_state.pop("backend_error", None)
    st.session_state.pop("current_session_id", None)
    st.session_state.pop("loaded_session_id", None)
    st.session_state.pop("message", None)


def disable_backend_mode(reason: str):
    st.session_state["backend_mode"] = False
    st.session_state["backend_error"] = reason
    logger.error(f"[backend]后端不可用，已降级为本地模式：{reason}")


def _normalize_backend_messages(messages: list[dict]) -> list[dict]:
    normalized = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        item = {"role": role, "content": content}
        refs = msg.get("references")
        if isinstance(refs, list) and refs:
            item["references"] = refs
        normalized.append(item)
    return normalized


def get_backend_client() -> BackendClient:
    client = st.session_state.get("backend_client")
    if not client or getattr(client, "base_url", "") != BACKEND_URL.rstrip("/"):
        st.session_state["backend_client"] = BackendClient(BACKEND_URL)
    return st.session_state["backend_client"]


def backend_list_sessions() -> list[dict]:
    return get_backend_client().list_sessions()


def backend_create_session(title: str = "新会话") -> dict:
    return get_backend_client().create_session(title=title)


def backend_create_chat_job(session_id: str, message: str) -> dict:
    return get_backend_client().create_chat_job(session_id=session_id, message=message)


def backend_get_chat_job(job_id: str) -> dict:
    return get_backend_client().get_chat_job(job_id)


def backend_list_active_chat_jobs() -> list[dict]:
    return get_backend_client().list_active_chat_jobs()


def backend_switch_session(session_id: str):
    detail = get_backend_client().get_session_detail(session_id)
    st.session_state["current_session_id"] = detail["session"]["id"]
    st.session_state["message"] = _normalize_backend_messages(detail.get("messages", []))
    st.session_state["loaded_session_id"] = detail["session"]["id"]


def backend_load_or_create_initial_session() -> dict:
    sessions = backend_list_sessions()
    if sessions:
        detail = get_backend_client().get_session_detail(sessions[0]["id"])
        st.session_state["message"] = _normalize_backend_messages(detail.get("messages", []))
        st.session_state["loaded_session_id"] = detail["session"]["id"]
        return detail["session"]

    created = backend_create_session()
    st.session_state["message"] = []
    st.session_state["loaded_session_id"] = created["id"]
    return created


def local_load_or_create_initial_session() -> dict:
    sessions = list_sessions()
    if sessions:
        st.session_state["message"] = sessions[0].get("messages", [])
        st.session_state["loaded_session_id"] = sessions[0]["id"]
        return sessions[0]
    created = create_session()
    st.session_state["message"] = []
    st.session_state["loaded_session_id"] = created["id"]
    return created


def persist_current_messages_local(session_id: str | None = None):
    session_id = session_id or st.session_state.get("current_session_id")
    if not session_id:
        return
    messages = st.session_state.get("message", [])
    save_session_messages(session_id, messages)


def local_switch_session(session_id: str):
    session = get_session(session_id)
    if session is None:
        session = create_session()
    st.session_state["current_session_id"] = session["id"]
    st.session_state["message"] = session.get("messages", [])
    st.session_state["loaded_session_id"] = session["id"]


def switch_session(session_id: str):
    if use_backend_api():
        backend_switch_session(session_id)
    else:
        local_switch_session(session_id)


def track_chat_job(job: dict):
    job_id = str(job.get("id", "")).strip()
    session_id = str(job.get("session_id", "")).strip()
    if not job_id or not session_id:
        return
    tracked = dict(st.session_state.get("tracked_chat_jobs", {}))
    tracked[job_id] = session_id
    st.session_state["tracked_chat_jobs"] = tracked


def refresh_backend_chat_jobs() -> tuple[list[dict], dict[str, dict]]:
    if not use_backend_api():
        return [], {}

    try:
        active_jobs = backend_list_active_chat_jobs()
    except BackendClientError as e:
        disable_backend_mode(str(e))
        return [], {}

    tracked = dict(st.session_state.get("tracked_chat_jobs", {}))
    for job in active_jobs:
        track_chat_job(job)
        tracked[str(job.get("id"))] = str(job.get("session_id"))

    active_ids = {str(job.get("id")) for job in active_jobs}
    current_id = st.session_state.get("current_session_id")
    should_reload_current = False

    for job_id, session_id in list(tracked.items()):
        if job_id in active_ids:
            continue
        try:
            job = backend_get_chat_job(job_id)
        except BackendClientError:
            tracked.pop(job_id, None)
            continue

        if job.get("status") in {"completed", "failed"}:
            if session_id == current_id:
                should_reload_current = True
            tracked.pop(job_id, None)

    st.session_state["tracked_chat_jobs"] = tracked

    if should_reload_current and current_id:
        try:
            backend_switch_session(current_id)
        except BackendClientError as e:
            disable_backend_mode(str(e))
            return [], {}

    active_by_session = {}
    for job in active_jobs:
        session_id = str(job.get("session_id", ""))
        if session_id:
            active_by_session[session_id] = job

    return active_jobs, active_by_session


def load_or_create_initial_session() -> dict:
    if use_backend_api():
        return backend_load_or_create_initial_session()
    return local_load_or_create_initial_session()


def ensure_current_session_messages_loaded():
    current_id = st.session_state.get("current_session_id")
    if not current_id or st.session_state.get("loaded_session_id") == current_id:
        return

    try:
        if use_backend_api():
            backend_switch_session(current_id)
        else:
            local_switch_session(current_id)
    except BackendClientError as e:
        disable_backend_mode(str(e))
        local_switch_session(current_id)


def get_current_conversation_title() -> str:
    for message in st.session_state.get("message", []):
        if message.get("role") == "user":
            content = str(message.get("content", "")).strip()
            if content:
                return content[:26] + ("..." if len(content) > 26 else "")
    return "新对话"


def render_fixed_topbar(disable_controls: bool = False):
    toggle_label = "‹" if st.session_state["sidebar_open"] else "›"
    with st.container(key="fixed_topbar"):
        col_toggle, col_title = st.columns([1, 10], vertical_alignment="center")
        with col_toggle:
            if st.button(toggle_label, key="toggle_sidebar", help="展开/收起侧边栏", disabled=disable_controls):
                st.session_state["sidebar_open"] = not st.session_state["sidebar_open"]
                st.rerun()
        with col_title:
            title = get_current_conversation_title()
            st.markdown(
                f"""
                <div class="topbar-title">{title}</div>
                <div class="topbar-subtitle">AI 生成可能有误，请核实</div>
                """,
                unsafe_allow_html=True,
            )


def render_assistant_references(message: dict):
    references = message.get("references") or []
    if not references:
        return

    with st.expander("参考来源", expanded=False):
        for ref in references:
            idx = ref.get("index", "-")
            source_name = ref.get("source_name", "未知来源")
            chunk_id = ref.get("chunk_id", "unknown")
            retrieval_hits = ref.get("retrieval_hits", "unknown")
            rerank_score = ref.get("rerank_score", "unknown")
            snippet = ref.get("snippet", "")

            st.markdown(f"**[{idx}] {source_name}**")
            st.caption(f"chunk_id: {chunk_id} | 召回通道: {retrieval_hits} | 重排分: {rerank_score}")
            st.write(snippet)
            st.divider()


def render_current_job_waiter(job_id: str, session_id: str):
    st.markdown(
        """
        <div class="thinking-indicator">
            <span class="thinking-mark"></span>
            <span>智能客服思考中...</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def queue_pending_prompt(prompt: str):
    text = str(prompt or "").strip()
    if not text:
        return
    st.session_state["pending_prompt"] = text
    st.session_state["response_in_progress"] = True


def queue_chat_input_prompt():
    queue_pending_prompt(st.session_state.get("chat_prompt_input", ""))


def render_quick_prompts():
    with st.container(key="quick_prompts"):
        cols = st.columns([1, 1, 1, 1])
        for idx, question in enumerate(QUICK_PROMPTS[:4]):
            with cols[idx]:
                if st.button(question, key=f"quick_prompt_{idx}", use_container_width=True):
                    queue_pending_prompt(question)
                    st.rerun()
        cols = st.columns([1, 1, 1])
        with cols[1]:
            if st.button(QUICK_PROMPTS[4], key="quick_prompt_4", use_container_width=True):
                queue_pending_prompt(QUICK_PROMPTS[4])
                st.rerun()


def rebuild_knowledge_base():
    with st.spinner("正在重建知识库，请稍候..."):
        if use_backend_api():
            get_backend_client().rebuild_knowledge_base()
        else:
            vs = VectorStoreService()
            vs.rebuild_document()
            reload_rag_service()
            st.session_state["agent"] = ReactAgent()
    st.success("知识库重建完成")


def reload_agent_runtime():
    if use_backend_api():
        get_backend_client().reload_agent()
    else:
        st.session_state["agent"] = ReactAgent()


def ensure_runtime_initialized():
    if use_backend_api():
        get_backend_client()
    else:
        if ALLOW_LOCAL_AGENT_FALLBACK and "agent" not in st.session_state:
            st.session_state["agent"] = ReactAgent()


restore_backend_mode_if_available()
ensure_runtime_initialized()

if "current_session_id" not in st.session_state or "message" not in st.session_state:
    try:
        initial_session = load_or_create_initial_session()
    except BackendClientError as e:
        disable_backend_mode(str(e))
        initial_session = local_load_or_create_initial_session()
    st.session_state["current_session_id"] = initial_session["id"]
    st.session_state.setdefault("message", [])


active_chat_jobs, active_jobs_by_session = refresh_backend_chat_jobs()
current_active_job = active_jobs_by_session.get(st.session_state.get("current_session_id", ""))
pending_prompt = st.session_state.pop("pending_prompt", None)
local_input_locked = bool(st.session_state.get("response_in_progress", False))
if use_backend_api():
    if local_input_locked and not pending_prompt:
        st.session_state["response_in_progress"] = False
    input_locked = False
else:
    input_locked = local_input_locked
    if input_locked and not pending_prompt:
        st.session_state["response_in_progress"] = False
        input_locked = False
chat_prompt = st.chat_input(
    "请输入你的问题...",
    key="chat_prompt_input",
    disabled=input_locked,
    on_submit=queue_chat_input_prompt,
)
prompt = pending_prompt or chat_prompt
is_local_generating_response = (bool(prompt) or input_locked) and not use_backend_api()
disable_current_session_mutation = is_local_generating_response or bool(current_active_job)
disable_admin_ops = is_local_generating_response or bool(active_chat_jobs)


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-avatar">洁</div>
            <div>
                <div class="sidebar-brand-title">洁境智顾</div>
                <div class="sidebar-brand-en">CleanEnv Agent</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 会话管理")
    st.caption("管理历史对话、知识库和当前 Agent 运行状态。")
    if USE_BACKEND_API and not use_backend_api():
        st.warning("后端当前不可用，已自动切换为本地模式。")
        backend_err = st.session_state.get("backend_error", "")
        if backend_err:
            st.caption(backend_err)
        if not ALLOW_LOCAL_AGENT_FALLBACK:
            st.caption("当前已禁用本地同步生成，请启动 FastAPI 后端后刷新页面。")
    if st.button("新建会话", use_container_width=True, disabled=is_local_generating_response):
        try:
            if use_backend_api():
                session = backend_create_session()
                st.session_state["current_session_id"] = session["id"]
                st.session_state["message"] = []
                st.session_state["loaded_session_id"] = session["id"]
            else:
                session = create_session()
                local_switch_session(session["id"])
            st.rerun()
        except BackendClientError as e:
            disable_backend_mode(str(e))
            st.rerun()

    try:
        sessions = backend_list_sessions() if use_backend_api() else list_sessions()
    except BackendClientError as e:
        disable_backend_mode(str(e))
        sessions = list_sessions()
        st.warning("后端会话读取失败，已切换本地模式。")
        st.caption(str(e))

    if not use_backend_api():
        sessions = []
        sessions = list_sessions()

    if sessions:
        current_id = st.session_state.get("current_session_id")
        options = [s["id"] for s in sessions]
        labels = {
            s["id"]: f"{s.get('title', '新会话')} ({s.get('updated_at', '')[:16].replace('T', ' ')})"
            for s in sessions
        }
        default_idx = options.index(current_id) if current_id in options else 0
        with st.expander("历史会话", expanded=True):
            selected_id = st.radio(
                "历史会话",
                options,
                index=default_idx,
                format_func=lambda sid: labels.get(sid, sid),
                label_visibility="collapsed",
                disabled=is_local_generating_response,
            )
        if selected_id != current_id:
            try:
                switch_session(selected_id)
                st.rerun()
            except BackendClientError as e:
                st.error(str(e))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("删除当前会话", use_container_width=True, disabled=disable_current_session_mutation):
            try:
                current_id = st.session_state.get("current_session_id")
                if current_id:
                    if use_backend_api():
                        get_backend_client().delete_session(current_id)
                    else:
                        delete_session(current_id)

                remaining = backend_list_sessions() if use_backend_api() else list_sessions()
                if remaining:
                    switch_session(remaining[0]["id"])
                else:
                    session = backend_create_session() if use_backend_api() else create_session()
                    if use_backend_api():
                        st.session_state["current_session_id"] = session["id"]
                        st.session_state["message"] = []
                        st.session_state["loaded_session_id"] = session["id"]
                    else:
                        local_switch_session(session["id"])
                st.rerun()
            except BackendClientError as e:
                disable_backend_mode(str(e))
                st.rerun()

    with col_b:
        if st.button("清空当前会话", use_container_width=True, disabled=disable_current_session_mutation):
            try:
                current_id = st.session_state.get("current_session_id")
                if current_id and use_backend_api():
                    get_backend_client().clear_session_messages(current_id)
                st.session_state["message"] = []
                if not use_backend_api():
                    persist_current_messages_local()
                st.rerun()
            except BackendClientError as e:
                disable_backend_mode(str(e))
                st.rerun()

    st.divider()
    st.caption("知识更新后可重建知识库；Prompt 或工具调整后可重载 Agent。")
    if st.button("重建知识库", use_container_width=True, disabled=disable_admin_ops):
        try:
            rebuild_knowledge_base()
        except BackendClientError as e:
            st.error(str(e))

    if st.button("重载Agent", use_container_width=True, disabled=disable_admin_ops):
        try:
            reload_agent_runtime()
            st.success("Agent已重载")
        except BackendClientError as e:
            st.error(str(e))


ensure_current_session_messages_loaded()
render_fixed_topbar(disable_controls=is_local_generating_response)

if not st.session_state["message"] and not prompt:
    with st.container(key="welcome_panel"):
        st.markdown(
            '<h2 class="welcome-title">有什么我能帮你的吗？</h2>',
            unsafe_allow_html=True,
        )
        render_quick_prompts()

for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_assistant_references(message)

if current_active_job and not prompt:
    render_current_job_waiter(
        str(current_active_job.get("id", "")),
        str(current_active_job.get("session_id", "")),
    )


if prompt:
    active_session_id = st.session_state.get("current_session_id", "")

    if use_backend_api():
        try:
            job = backend_create_chat_job(active_session_id, prompt)
            track_chat_job(job)
            st.session_state["response_in_progress"] = False
            backend_switch_session(active_session_id)
        except BackendClientError as e:
            disable_backend_mode(str(e))
            st.session_state["response_in_progress"] = False
        st.rerun()

    if not ALLOW_LOCAL_AGENT_FALLBACK:
        st.error("企业版后端未连接，请先启动 FastAPI 后端：uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
        st.stop()

    st.session_state["response_in_progress"] = True
    user_message = {"role": "user", "content": prompt}
    st.session_state["message"].append(user_message)
    persist_current_messages_local(active_session_id)

    with st.chat_message("user"):
        st.write(prompt)

    try:
        with st.spinner("智能客服思考中..."):
            with request_context() as request_id:
                logger.info(f"[chat]开始处理用户请求，请求ID：{request_id}")

                clear_last_rag_references()
                chat_history = [
                    {"role": msg.get("role"), "content": msg.get("content")}
                    for msg in st.session_state["message"]
                    if msg.get("role") in {"user", "assistant"}
                ]
                res_stream = st.session_state["agent"].execute_stream(chat_history)
                response_chunks = []

                def capture(generator):
                    for chunk in generator:
                        response_chunks.append(chunk)
                        for char in chunk:
                            time.sleep(0.01)
                            yield char

                with st.chat_message("assistant"):
                    rendered_text = st.write_stream(capture(res_stream))

                full_response = "".join(response_chunks).strip()
                if not full_response and isinstance(rendered_text, str):
                    full_response = rendered_text.strip()

                references = get_last_rag_references()
                assistant_message = {"role": "assistant", "content": full_response}
                if references:
                    assistant_message["references"] = references
                st.session_state["message"].append(assistant_message)
                persist_current_messages_local(active_session_id)

                logger.info(f"[chat]请求处理完成，请求ID：{request_id}")
    finally:
        st.session_state["response_in_progress"] = False

    st.rerun()
