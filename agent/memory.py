import datetime
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

Base = declarative_base()
logging.getLogger("sqlalchemy").setLevel(logging.ERROR)


class AgentConversation(Base):
    __tablename__ = "agent_history"
    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger, nullable=False)
    tool = Column(String)
    status = Column(String)
    attempt = Column(String)
    stdout = Column(Text)
    stderr = Column(Text)
    total_tokens = Column(Integer)
    prompt_tokens = Column(Integer)
    response_tokens = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(Integer)


class AgentMemory:
    def __init__(self):
        self.database_url = "sqlite:///agent_memory.db"
        self.engine = create_engine(self.database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_conversation_memory(
        self,
        user_id,
        run_id,
        previous_subtask_tool,
        previous_subtask_result,
        previous_subtask_attempt,
        previous_subtask_output,
        previous_subtask_errors,
        total_tokens,
        prompt_tokens,
        response_tokens,
    ):
        session = self.Session()
        conversation = AgentConversation(
            user_id=user_id,
            run_id=run_id,
            tool=str(previous_subtask_tool),
            status=str(previous_subtask_result),
            attempt=str(previous_subtask_attempt),
            stdout=str(previous_subtask_output),
            stderr=str(previous_subtask_errors),
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
        )
        session.add(conversation)
        session.commit()
        session.close()

    def get_conversation_memory(self, run_id):
        session = self.Session()
        short_term_conversations = (
            session.query(AgentConversation)
            .filter_by(run_id=run_id)
            .order_by(AgentConversation.created_at.desc())
            .limit(50)
            .all()
        )
        short_term_memories = []
        for conversation in reversed(short_term_conversations):
            short_term_memories.append(
                {
                    "tool": conversation.tool,
                    "status": conversation.status,
                    "attempt": conversation.attempt,
                    "stdout": conversation.stdout,
                    "stderr": conversation.stderr,
                    "total_tokens": conversation.total_tokens,
                    "prompt_tokens": conversation.prompt_tokens,
                    "response_tokens": conversation.response_tokens,
                }
            )

        max_output_chars = 1000

        full_output_mems = ""
        for idx, item in enumerate(short_term_memories):
            tool = item.get("tool", "unknown")
            attempt = item.get("attempt", "")
            stdout = item.get("stdout", "")
            stderr = item.get("stderr", "")
            if len(stdout) > max_output_chars:
                stdout = stdout[:max_output_chars] + f"\n  ... (truncated, {len(stdout)} chars total)"
            if len(stderr) > max_output_chars:
                stderr = stderr[:max_output_chars] + f"\n  ... (truncated, {len(stderr)} chars total)"
            parts = [f"Step {idx + 1}: [{tool}] {attempt}"]
            if stdout:
                parts.append(f"  Output: {stdout}")
            if stderr:
                parts.append(f"  Error: {stderr}")
            full_output_mems += "\n".join(parts) + "\n"

        return full_output_mems
