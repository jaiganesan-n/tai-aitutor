"""The Gradio AI-tutor app, rebuilt on the package — the reference for the
`ai-tutor-gradio-lesson` repo rewrite (and the Section 16 certification skeleton).

What the old app.py did with `OpenAIAgent.from_tools` + `RetrieverTool` +
`ChatSummaryMemoryBuffer` (166 lines of framework), this does with parts the
student built: a `Chat` with summary memory, one retrieval tool, streamed to
Gradio. Models are NOT hard-coded: the provider dropdown drives `configure`.

Run locally:  pip install "tai-aitutor[gemini,openai,anthropic,rag,data]" gradio
              python examples/gradio_tutor.py
"""

from tai_aitutor import Chat, configure, get_collection, make_retrieval_tool, setup_notebook
from tai_aitutor.datasets import prebuilt_chroma

SYSTEM_PROMPT = (
    "You are an AI teacher answering questions from students of an applied AI course "
    "on Large Language Models and Retrieval Augmented Generation. Ground every answer "
    "in the course knowledge base via the search tool; politely decline questions "
    "unrelated to AI, machine learning, or the course."
)


def build_chat(provider: str) -> Chat:
    # ⚠ the hosted store is embedded with OpenAI text-embedding-3-small — the query-time
    # embedder must match (this is the coupling the course plan flags; the Gemini-embedded
    # store replaces it with the org data migration).
    configure(provider=provider, embed_provider="openai")
    store_path = prebuilt_chroma()
    col = get_collection("ai_tutor_knowledge", path=str(store_path))
    return Chat(
        system=SYSTEM_PROMPT,
        tools=[make_retrieval_tool(col, top_k=5)],
        history="summary",  # the hand-rolled summary memory from the memory lesson
    )


def main() -> None:
    import gradio as gr

    setup_notebook(required_keys=("OPENAI_API_KEY",))  # + the chat provider's key
    state: dict[str, Chat] = {}

    def respond(message, history, provider):
        if state.get("provider") != provider:
            state["chat"] = build_chat(provider)
            state["provider"] = provider
        chat: Chat = state["chat"]
        partial = ""
        for event in chat.ask_stream(message):
            if event.type == "tool_call":
                yield partial + f"\n\n_searching: {event.arguments.get('query', '')}…_"
            elif event.type == "text":
                partial = event.text
                yield partial

    demo = gr.ChatInterface(
        respond,
        additional_inputs=[
            gr.Dropdown(["gemini", "openai", "anthropic"], value="gemini", label="Provider")
        ],
        title="AI Tutor 🧑‍🏫",
        description="Ask about the course material. Answers are grounded in the knowledge base.",
        type="messages",
    )
    demo.launch(share=False, debug=False)  # deliberately NOT share=True/debug=True


if __name__ == "__main__":
    main()
