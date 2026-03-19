from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OpenaiChatMessage(BaseModel):
    role: str
    content: Any
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class MiddlemanSettings(BaseModel):
    n: int = 1
    model: str = "openai/gpt-5.2"
    temp: float = 0.0
    max_tokens: int = 1024
    stop: List[str] = Field(default_factory=list)


class MiddlemanOutput(BaseModel):
    completion: str = ""
    function_call: Optional[Dict[str, Any]] = None


class MiddlemanResult(BaseModel):
    outputs: Optional[List[MiddlemanOutput]] = None
    model: Optional[str] = None
    usage: Dict[str, Any] = Field(default_factory=dict)


class RatingOption(BaseModel):
    action: str
    fixedRating: Optional[float] = None


class RatedOption(BaseModel):
    action: str
    fixedRating: Optional[float] = None
    rating: Optional[float] = None


class ExecResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exitStatus: int = 0


class ScoreResult(BaseModel):
    status: str = "noScore"
    score: Optional[float] = None
    message: Dict[str, Any] = Field(default_factory=dict)
    execResult: ExecResult = Field(default_factory=ExecResult)


class ScoreLogEntry(BaseModel):
    status: str = "noScore"
    score: Optional[float] = None
    message: Dict[str, Any] = Field(default_factory=dict)


class UsageSnapshot(BaseModel):
    tokens: int = 0
    total_seconds: int = 0


class UsageLimits(BaseModel):
    tokens: int = 500000
    total_seconds: int = 7200


class UsageInfo(BaseModel):
    usage: UsageSnapshot = Field(default_factory=UsageSnapshot)
    usageLimits: UsageLimits = Field(default_factory=UsageLimits)


class TaskScoring(BaseModel):
    intermediate: bool = False


class Task(BaseModel):
    instructions: str
    scoring: TaskScoring = Field(default_factory=TaskScoring)
