"""Conversation and tool loops: memory is the message list you choose to resend.

Built in: "Enabling Conversational Memory" (Section 9) — the lesson hand-rolls
full history, a sliding window, and summarize-old-turns before importing them
from here — and the Web Search / agents lessons for :class:`ToolLoop`.

Replaces ``index.as_chat_engine(chat_mode=..., memory=...)``, ``FunctionAgent``,
``ReActAgent``, ``AgentStream`` / ``ToolCallResult``, workflow ``Context``, and
the Gradio app's ``ChatSummaryMemoryBuffer``. There is no engine object: a chat
is a list of messages, a loop over :func:`~tai_aitutor.llm.chat_completion`,
and three visible strategies for which part of the list gets resent.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from . import llm as _llm
from .errors import TaiAitutorError
from .llm import ToolCall, Usage
from .tokens import n_tokens
from .tools import Tool, render_tool_result

__all__ = ["ChatEvent", "ToolLoop", "Chat"]

HISTORY_MODES = ("full", "window", "summary")


@dataclass(frozen=True)
class ChatEvent:
    """One thing that happened during a turn (replaces ``AgentStream``/``ToolCallResult``).

    ``type`` is ``"text"`` (final answer text), ``"tool_call"`` (the model asked
    to run a tool), or ``"tool_result"`` (what the tool returned).
    """

    type: str
    text: str = ""
    name: str = ""
    arguments: dict = field(default_factory=dict)
    result: str = ""


def _agent_events(
    messages: list[dict],
    tools: list[Tool],
    *,
    system: str | None,
    model: str | None,
    provider: str | None,
    max_iters: int,
    max_tokens: int | None,
    usage_sink: list[Usage] | None = None,
) -> Iterator[ChatEvent]:
    """The bare agent loop: call the model; run requested tools; repeat until text.

    Appends every assistant/tool message to ``messages`` in place (the visible
    transcript), yields events as they happen, and ends with a ``"text"`` event.
    """
    by_name = {t.name: t for t in tools}
    for _ in range(max_iters):
        completion = _llm.chat_completion(
            messages,
            tools=tools or None,
            system=system,
            model=model,
            provider=provider,
            max_tokens=max_tokens,
        )
        if usage_sink is not None:
            usage_sink.append(completion.usage)

        if not completion.tool_calls:
            messages.append({"role": "assistant", "content": completion.text})
            yield ChatEvent(type="text", text=completion.text)
            return

        messages.append(
            {
                "role": "assistant",
                "content": completion.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in completion.tool_calls
                ],
            }
        )
        for call in completion.tool_calls:
            yield ChatEvent(type="tool_call", name=call.name, arguments=call.arguments)
            result_text = _run_tool(by_name, call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": result_text,
                }
            )
            yield ChatEvent(type="tool_result", name=call.name, result=result_text)

    raise TaiAitutorError(
        f"Tool loop did not settle after {max_iters} iterations — raise max_iters "
        "or check the tools for loops (a tool erroring repeatedly does this)."
    )


def _run_tool(by_name: dict[str, Tool], call: ToolCall) -> str:
    """Run one tool call; errors go back to the model as text (it can recover)."""
    tool = by_name.get(call.name)
    if tool is None:
        return f"ERROR: unknown tool {call.name!r}. Available: {sorted(by_name)}"
    try:
        return render_tool_result(tool.fn(**call.arguments))
    except Exception as exc:  # noqa: BLE001 — the model gets the error and can retry
        return f"ERROR: {type(exc).__name__}: {exc}"


class ToolLoop:
    """The bare native tool-calling loop, stateless per call (teachable standalone).

    Replaces ``FunctionAgent`` / ``ReActAgent``: no agent object state — a
    question goes in, tools run as the model requests them, an answer comes out.

    >>> loop = ToolLoop(tools=[tool(search_web)], system="Ground answers in search.")
    >>> loop.run("What changed in Python 3.14?")
    """

    def __init__(
        self,
        tools: list[Tool] = (),
        system: str | None = None,
        *,
        model: str | None = None,
        provider: str | None = None,
        max_iters: int = 6,
        max_tokens: int | None = None,
    ):
        self.tools = list(tools)
        self.system = system
        self.model = model
        self.provider = provider
        self.max_iters = max_iters
        self.max_tokens = max_tokens

    def run(self, user_message: str) -> str:
        """One question → final answer (tools run in between as needed)."""
        final = ""
        for event in self.run_events(user_message):
            if event.type == "text":
                final = event.text
        return final

    def run_events(self, user_message: str) -> Iterator[ChatEvent]:
        """Like :meth:`run`, but yields every tool call/result as it happens."""
        messages = [{"role": "user", "content": user_message}]
        yield from _agent_events(
            messages,
            self.tools,
            system=self.system,
            model=self.model,
            provider=self.provider,
            max_iters=self.max_iters,
            max_tokens=self.max_tokens,
        )


class Chat:
    """A conversation: the visible message list plus a strategy for resending it.

    ``history=`` picks the memory strategy the lesson builds by hand first:

    - ``"full"`` — resend everything (with prompt caching this is often the
      cheapest strategy in production, which Section 13 measures);
    - ``"window"`` — resend only the last ``window_turns`` turns;
    - ``"summary"`` — when the resent history would exceed
      ``summarize_after_tokens``, summarize older turns once and resend the
      summary plus recent turns (the compaction trade-off Section 13 measures).

    ``self.messages`` is always the complete transcript, tool traffic included
    — what got *sent* last turn is ``self.last_context`` (inspect both; the
    difference IS the lesson).

    >>> chat = Chat(system="You are the course tutor.",
    ...             tools=[make_retrieval_tool(col)], history="window")
    >>> chat.ask("What is RAG?")
    >>> chat.ask("And how does it differ from fine-tuning?")
    """

    def __init__(
        self,
        system: str | None = None,
        tools: list[Tool] = (),
        history: str = "full",
        *,
        window_turns: int = 8,
        summarize_after_tokens: int = 8000,
        model: str | None = None,
        provider: str | None = None,
        max_iters: int = 6,
        max_tokens: int | None = None,
    ):
        if history not in HISTORY_MODES:
            raise TaiAitutorError(f"history must be one of {HISTORY_MODES}, got {history!r}")
        self.system = system
        self.tools = list(tools)
        self.history = history
        self.window_turns = window_turns
        self.summarize_after_tokens = summarize_after_tokens
        self.model = model
        self.provider = provider
        self.max_iters = max_iters
        self.max_tokens = max_tokens

        self._turns: list[list[dict]] = []  # each turn: [user, ...tool traffic..., assistant]
        self._summary: str = ""
        self._summarized_upto: int = 0
        self.last_context: list[dict] = []
        self.usage = Usage(0, 0)

    # -- transcript ---------------------------------------------------------

    @property
    def messages(self) -> list[dict]:
        """The complete transcript (every turn, tool traffic included)."""
        return [message for turn in self._turns for message in turn]

    def reset(self) -> None:
        self._turns, self._summary, self._summarized_upto = [], "", 0
        self.last_context, self.usage = [], Usage(0, 0)

    # -- memory strategies (the lesson's three, verbatim) -------------------

    def _context_turns(self) -> list[list[dict]]:
        if self.history == "full":
            return self._turns
        if self.history == "window":
            return self._turns[-self.window_turns :]
        # summary mode: maybe fold older turns into the running summary first
        self._maybe_summarize()
        return self._turns[self._summarized_upto :]

    def _context_messages(self) -> list[dict]:
        return [message for turn in self._context_turns() for message in turn]

    def _system_with_summary(self) -> str | None:
        if self.history == "summary" and self._summary:
            base = self.system or ""
            return f"{base}\n\nSummary of the conversation so far:\n{self._summary}".strip()
        return self.system

    def _maybe_summarize(self) -> None:
        candidate = self._turns[self._summarized_upto :]
        if len(candidate) <= self.window_turns:
            return
        tokens = sum(
            n_tokens(str(message.get("content") or "")) for turn in candidate for message in turn
        )
        if tokens <= self.summarize_after_tokens:
            return
        to_fold = self._turns[self._summarized_upto : -self.window_turns]
        transcript = "\n".join(
            f"{message['role']}: {message.get('content') or ''}"
            for turn in to_fold
            for message in turn
            if message["role"] in ("user", "assistant") and message.get("content")
        )
        previous = f"Previous summary:\n{self._summary}\n\n" if self._summary else ""
        self._summary = _llm.generate(
            f"{previous}Conversation to fold into the summary:\n{transcript}\n\n"
            "Write a compact summary preserving every fact, preference, and open "
            "question a tutor would need later.",
            model=self.model,
            provider=self.provider,
        ).strip()
        self._summarized_upto = len(self._turns) - self.window_turns

    # -- asking -------------------------------------------------------------

    def ask(self, user_message: str) -> str:
        """Send one user message; run tools as requested; return the answer text."""
        final = ""
        for event in self.ask_stream(user_message):
            if event.type == "text":
                final = event.text
        return final

    def ask_stream(self, user_message: str) -> Iterator[ChatEvent]:
        """Like :meth:`ask`, but yields tool calls/results/answer as events."""
        context = self._context_messages()  # strategy applies BEFORE the new turn
        turn: list[dict] = [{"role": "user", "content": user_message}]
        self._turns.append(turn)
        working = context + list(turn)
        self.last_context = working

        before = len(working)
        usage_sink: list[Usage] = []
        try:
            yield from _agent_events(
                working,
                self.tools,
                system=self._system_with_summary(),
                model=self.model,
                provider=self.provider,
                max_iters=self.max_iters,
                max_tokens=self.max_tokens,
                usage_sink=usage_sink,
            )
        finally:
            turn.extend(working[before:])  # record everything the loop appended
            self.usage = Usage(
                self.usage.input_tokens + sum(u.input_tokens for u in usage_sink),
                self.usage.output_tokens + sum(u.output_tokens for u in usage_sink),
            )
