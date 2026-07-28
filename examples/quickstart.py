"""The whole course pipeline, end to end, on the course data.

Needs: pip install "tai-aitutor[gemini,rag,data]" and GOOGLE_API_KEY in .env.
"""

from tai_aitutor import (
    answer,
    configure,
    get_collection,
    ingest,
    mini_articles,
    setup_notebook,
    show_answer,
)

setup_notebook(required_keys=("GOOGLE_API_KEY",))
configure(provider="gemini")

col = get_collection("mini_articles", path="./db")
if col.count() == 0:
    stats = ingest(mini_articles(), col, chunk_size=512, chunk_overlap=128)
    print(stats)

show_answer(answer("What is the difference between RAG and fine-tuning?", col, top_k=5))
