from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    title: str = Field(default="新会话", max_length=120)


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    references: list[dict] = Field(default_factory=list)
    created_at: str


class SessionDetailOut(BaseModel):
    session: SessionOut
    messages: list[MessageOut]


class ChatStreamRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=4000)


class ChatJobRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=4000)


class ChatJobOut(BaseModel):
    id: str
    session_id: str
    status: str
    prompt: str
    response_content: str = ""
    error_message: str = ""
    references: list[dict] = Field(default_factory=list)
    user_message_id: int | None = None
    assistant_message_id: int | None = None
    created_at: str
    updated_at: str
    started_at: str = ""
    completed_at: str = ""


class OperationOut(BaseModel):
    ok: bool
    message: str
