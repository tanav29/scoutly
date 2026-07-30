"""Provider boundary for grounded generation.

OpenAI is used when configured; the deterministic extractive fallback keeps
local development and tests usable without credentials or network access.
"""

from __future__ import annotations

import os
from typing import Iterable


def grounded_completion(question: str, evidence: Iterable[str]) -> str:
    evidence = list(evidence)
    context = "\n\n".join(evidence)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("SCOUTLY_CHAT_MODEL", "gpt-4o-mini"),
                temperature=0,
                messages=[
                    {"role": "system", "content": "Answer only from the supplied evidence. If it is insufficient, say so explicitly. Cite evidence with its bracket number."},
                    {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"},
                ],
            )
            return response.choices[0].message.content or "I couldn’t find enough support in the selected sources."
        except Exception:
            # A provider outage must not turn into an ungrounded answer.
            pass
    return "Based on the selected sources:\n\n" + context + "\n\nCitations: " + ", ".join(f"[{i + 1}]" for i, _ in enumerate(evidence))
