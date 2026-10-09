class AnswerPlanner:
    """Build concise, deterministic task instructions for grounded generation."""

    TASKS = {
        "direct_question": (
            "Answer the user's specific question directly and concisely. Do not answer a different question."
        ),
        "document_search": (
            "You are helping the user locate a document. Identify the best matching document from the supplied evidence and summarize only information supported by the evidence. Include the exact filename, exam and year, organizing institute, relevant pages, and visible topics or questions when those details appear in the evidence. Do not treat a broad document query as a numbered-question lookup. State when a requested detail is not available. Cite factual claims."
        ),
        "document_summary": (
            "Summarize the supplied document evidence. Combine distinct chunks, remove repetition, and organize the answer into useful sections. Include document identity, exam or year, institute, and visible topics only when the supplied evidence supports them. Cite each factual claim and say when the evidence is incomplete."
        ),
        "question_lookup": (
            "Find the requested numbered question in the supplied evidence. Quote or summarize only that exact question and identify its document and page. If that number is not present, say so. Do not substitute a different question with a similar number or topic. Cite the matched question."
        ),
        "question_answering": (
            "Answer the exact numbered question using only its supplied evidence. Explain reasoning only when the evidence supports it. If the matching question or enough information to answer it is absent, say so; do not answer a different question. Cite the evidence used."
        ),
        "memory_summary": (
            "Synthesize the distinct retrieved memories into a useful, concise summary. Group related points by topic, remove repeated chunks, cite each important claim, and explicitly state when the retrieved evidence is incomplete. Do not simply repeat individual chunks."
        ),
        "summarization": (
            "Synthesize the distinct points in the supplied memories into a concise summary. Group related points by topic or source, remove repetition, and cite each supported claim. Do not merely list or restate chunks."
        ),
        "comparison": (
            "Compare only the sources represented in the supplied evidence. Structure the answer as Similarities, Differences, and Important changes where supported. Say when evidence for one side is missing; do not invent changes."
        ),
        "temporal_summary": (
            "Summarize the supplied memories recorded within the requested date range, in chronological order. Describe them as recorded documents or memories; do not infer unrecorded personal activity. Do not claim that a listed item is the only memory unless the supplied evidence establishes that. Keep the summary concise and cite the supporting entries."
        ),
        "list_extraction": (
            "Extract the requested items as a concise list. Include only items explicitly supported by the supplied evidence and cite them."
        ),
        "explanation": (
            "Explain the requested concept using the supplied evidence. Keep the explanation focused on the user's question and cite supporting claims."
        ),
    }

    def plan(self, understanding: dict) -> str:
        intent = understanding.get("intent", "direct_question")
        task = self.TASKS.get(intent, self.TASKS["direct_question"])
        if understanding.get("question_number") is not None and intent in {
            "question_lookup", "question_answering"
        }:
            task += (
                f" Focus specifically on question {understanding['question_number']}; "
                "do not summarize unrelated questions."
            )
        time_range = understanding.get("time_range")
        if intent == "temporal_summary" and time_range:
            task += f" Requested range: {time_range['start']} through {time_range['end']}."
        return task


CONVERSATIONAL_SYSTEM_PROMPT = (
    "You are MemoraAI, a friendly local assistant. Respond naturally and briefly to "
    "greetings and conversational messages. Do not claim to have searched the user's "
    "documents or know personal facts that were not provided."
)
