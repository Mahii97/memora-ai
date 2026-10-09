import calendar
import re
from datetime import date, datetime, timedelta


STOP_WORDS = {
    "a", "about", "across", "an", "and", "are", "as", "at", "can", "could",
    "did", "do", "does", "find", "for", "from", "give", "hey", "hi", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "show",
    "some", "tell", "that", "the", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with", "would", "you", "your",
}
FILLER_WORDS = {
    "can", "could", "please", "tell", "show", "me", "find", "where", "is",
    "the", "my", "this", "that", "about", "give", "look", "up", "locate",
}
GENERIC_WORDS = {
    "date", "paper", "document", "documents", "file", "files", "page", "pages",
    "memory", "memories", "data",
}

CONVERSATIONAL_PATTERNS = (
    r"^(?:hi|hello|hey|good morning|good afternoon|good evening|how are you|"
    r"thanks|thank you|thx|good night)[.!? ]*$",
)
DOCUMENT_PATTERNS = (
    r"\bfind\b.*\b(?:paper|pdf|document|file|notes|project|resume|report)\b",
    r"\b(?:which|what)\s+document\b",
    r"\bshow\b.*\b(?:document|file|paper|pdf|project|notes)\b",
    r"\b(?:list|find|show)\s+(?:my\s+)?(?:documents|files|papers)\b",
)
DOCUMENT_SUMMARY_PATTERNS = (
    r"\b(?:what is|what's)\s+(?:this|the|my)?\s*(?:paper|document|file)\s+about\b",
    r"\b(?:overview|summary)\s+(?:of|for)\s+(?:this|the|my)?\s*(?:paper|document|file)\b",
)
SUMMARY_WORDS = {"summarize", "summarise", "summary", "overview", "summarization"}
MEMORY_SUMMARY_PATTERNS = (
    r"\bwhat\s+do\s+you\s+remember\b",
    r"\bmost\s+relevant\s+memories\b",
    r"\bmy\s+(?:memories|projects|work|notes)\b.*\b(?:summarize|summary|remember|recall)\b",
)
COMPARISON_WORDS = {"compare", "comparison", "difference", "differences", "changed", "versus", "vs"}
LIST_WORDS = {"list", "enumerate", "extract", "items"}
EXPLANATION_WORDS = {"explain", "why", "how"}
MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): index for index, name in enumerate(calendar.month_abbr) if name})


def _date_range(question: str, today: date) -> dict | None:
    text = question.casefold()
    if re.search(r"\b(?:today|this day)\b", text):
        start = end = today
    elif re.search(r"\byesterday\b", text):
        start = end = today - timedelta(days=1)
    elif re.search(r"\b(?:last|past) week\b", text):
        week_start = today - timedelta(days=today.weekday())
        end = week_start - timedelta(days=1)
        start = end - timedelta(days=6)
    elif re.search(r"\b(?:last|past) month\b", text):
        end = today.replace(day=1) - timedelta(days=1)
        start = end.replace(day=1)
    elif re.search(r"\bthis month\b", text):
        start = today.replace(day=1)
        end = today
    else:
        month_year = re.search(r"\b([a-z]+)\s+(20\d{2})\b", text)
        if month_year and month_year.group(1) in MONTHS:
            year = int(month_year.group(2))
            month = MONTHS[month_year.group(1)]
            return {
                "start": date(year, month, 1).isoformat(),
                "end": date(year, month, calendar.monthrange(year, month)[1]).isoformat(),
            }
        month_range = re.search(
            r"\b(?:between|from)\s+([a-z]+)(?:\s+(20\d{2}))?\s+(?:and|to|through)\s+([a-z]+)(?:\s+(20\d{2}))?\b",
            text,
        )
        if month_range and month_range.group(1) in MONTHS and month_range.group(3) in MONTHS:
            start_month = MONTHS[month_range.group(1)]
            end_month = MONTHS[month_range.group(3)]
            start_year = int(month_range.group(2) or today.year)
            end_year = int(month_range.group(4) or start_year + (1 if end_month < start_month else 0))
            start = date(start_year, start_month, 1)
            end = date(end_year, end_month, calendar.monthrange(end_year, end_month)[1])
            return {"start": start.isoformat(), "end": end.isoformat()}
        range_match = re.search(
            r"\bbetween\s+([a-z]+\s+\d{1,2})\s+(?:and|to|through)\s+([a-z]+\s+\d{1,2})\b",
            text,
        )
        if range_match:
            try:
                start = datetime.strptime(
                    f"{range_match.group(1)} {today.year}", "%B %d %Y"
                ).date()
                end = datetime.strptime(
                    f"{range_match.group(2)} {today.year}", "%B %d %Y"
                ).date()
            except ValueError:
                return None
        else:
            iso_range = re.search(
                r"\bbetween\s+(\d{4}-\d{2}-\d{2})\s+(?:and|to|through)\s+(\d{4}-\d{2}-\d{2})\b",
                text,
            )
            iso_date = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
            month_day = re.search(r"\b([a-z]+)\s+(\d{1,2})\b", text)
            try:
                if iso_range:
                    start = date.fromisoformat(iso_range.group(1))
                    end = date.fromisoformat(iso_range.group(2))
                elif iso_date:
                    start = end = date.fromisoformat(iso_date.group(1))
                elif month_day and month_day.group(1) in MONTHS:
                    start = end = date(
                        today.year, MONTHS[month_day.group(1)], int(month_day.group(2))
                    )
                else:
                    return None
            except ValueError:
                return None
    if start > end:
        start, end = end, start
    return {"start": start.isoformat(), "end": end.isoformat()}


class QueryUnderstanding:
    """Deterministic, local query analysis; never generates unsupported facts."""

    def analyze(self, question: str, today: date | None = None) -> dict:
        original = question.strip()
        normalized = re.sub(r"\s+", " ", original.casefold()).strip()
        words = re.findall(r"[\w]+", normalized, flags=re.UNICODE)
        has_domain_context = bool(re.search(r"\b(?:gate|paper|pdf|document|exam)\b", normalized))
        likely_date_typo = bool(
            "date" in words and has_domain_context
            and re.search(r"\b(?:ds|da|data\s+science)\b", normalized)
            and not re.search(r"\b(?:today|yesterday|last|past|this|september|october|november|december|january|february|march|april|may|june|july|august)\b", normalized)
        )
        normalized_for_intent = normalized
        normalizations = []
        if likely_date_typo:
            normalized_for_intent = re.sub(r"\bdate\b", "data", normalized_for_intent)
            normalizations.append({"from": "date", "to": "data", "reason": "likely data science paper typo"})
        words = re.findall(r"[\w]+", normalized_for_intent, flags=re.UNICODE)
        keywords = list(dict.fromkeys(
            word for word in words if word not in STOP_WORDS | FILLER_WORDS | GENERIC_WORDS
        ))

        question_match = re.search(
            r"\b(?:question|q)\s*\.?\s*(\d{1,4})\b|\b(\d{1,4})\s+(?:question|paper question)\b",
            normalized,
        )
        question_number = next((int(value) for value in question_match.groups() if value), None) if question_match else None
        entities = []
        if re.search(r"\bgate\b", normalized):
            entities.append("GATE")
        if re.search(r"\b(?:ds|da|data science)\b", normalized):
            entities.append("Data Science")
        if question_number is not None:
            entities.append(f"Question {question_number}")

        is_conversation = any(re.fullmatch(pattern, normalized) for pattern in CONVERSATIONAL_PATTERNS)
        is_comparison = bool(COMPARISON_WORDS.intersection(words))
        time_range = _date_range(original, today or date.today())
        is_temporal = time_range is not None
        is_document_search = any(re.search(pattern, normalized_for_intent) for pattern in DOCUMENT_PATTERNS)
        has_domain_entity = bool(re.search(r"\b(?:gate|ds|da|data\s+science)\b", normalized_for_intent))
        has_document_noun = bool(re.search(r"\b(?:paper|pdf|document|file|exam)\b", normalized_for_intent))
        is_memory_summary = any(re.search(pattern, normalized_for_intent) for pattern in MEMORY_SUMMARY_PATTERNS)
        is_document_summary = (
            has_document_noun
            and (
                bool(SUMMARY_WORDS.intersection(words))
                or any(re.search(pattern, normalized_for_intent) for pattern in DOCUMENT_SUMMARY_PATTERNS)
            )
        )
        answer_cue = bool(re.search(r"\b(?:solve|answer|solution|work out)\b", normalized_for_intent))
        is_question_answering = question_match is not None and answer_cue
        is_question_lookup = question_match is not None and not is_question_answering
        is_memory_summary = is_memory_summary or (
            bool(SUMMARY_WORDS.intersection(words))
            and bool(re.search(r"\b(?:memory|memories|remember|projects?)\b", normalized_for_intent))
        )
        is_document_discovery = is_document_search or (
            has_domain_entity and has_document_noun
        ) or (
            bool(re.fullmatch(r"(?:gate\s+)?(?:ds|da|data\s+science)(?:\s+(?:paper|data))?", normalized_for_intent))
        )

        if is_conversation:
            intent, action = "conversational", "respond_conversationally"
        elif is_comparison:
            intent, action = "comparison", "compare_sources"
        elif is_temporal:
            intent, action = "temporal_summary", "summarize_activity"
        elif is_question_answering:
            intent, action = "question_answering", "answer_numbered_question"
        elif is_question_lookup:
            intent, action = "question_lookup", "locate_numbered_question"
        elif is_document_summary:
            intent, action = "document_summary", "summarize_document"
        elif is_memory_summary:
            intent, action = "memory_summary", "summarize_memories"
        elif SUMMARY_WORDS.intersection(words):
            intent, action = "summarization", "summarize_evidence"
        elif LIST_WORDS.intersection(words):
            intent, action = "list_extraction", "extract_supported_items"
        elif EXPLANATION_WORDS.intersection(words):
            intent, action = "explanation", "explain_from_evidence"
        elif is_document_discovery:
            intent, action = "document_search", "find_document"
        else:
            intent, action = "direct_question", "answer_question"

        question_type = intent
        retrieval_terms = []
        if question_number is not None:
            retrieval_terms.extend(
                [f"Question {question_number}", f"Q.{question_number}", f"Q{question_number}"]
            )
        if "gate" in words and ("ds" in words or "da" in words or "science" in words):
            retrieval_terms.extend(
                ["GATE DA", "Data Science", "Data Science and Artificial Intelligence"]
            )
        elif has_domain_entity and has_document_noun:
            retrieval_terms.extend(["Data Science", "Data Science and Artificial Intelligence"])
        query_for_retrieval = normalized_for_intent
        query_for_retrieval = re.sub(r"\b(?:can you|could you|please|tell me|show me|about)\b", " ", query_for_retrieval)
        query_for_retrieval = re.sub(r"\bdate\b", "data", query_for_retrieval) if likely_date_typo else query_for_retrieval
        if has_domain_context:
            query_for_retrieval = re.sub(r"\bds\b", "data science", query_for_retrieval)
            if re.search(r"\bda\b", query_for_retrieval):
                query_for_retrieval = re.sub(r"\bda\b", "data science artificial intelligence", query_for_retrieval)
        query_terms = [word for word in re.findall(r"[\w]+", query_for_retrieval) if word not in STOP_WORDS | FILLER_WORDS | GENERIC_WORDS]
        retrieval_query = " ".join(dict.fromkeys(query_terms)) or query_for_retrieval.strip()
        if question_number is not None:
            retrieval_terms.extend([f"Question {question_number}", f"Q.{question_number}", f"Q{question_number}"])
        if retrieval_terms:
            retrieval_query = f"{retrieval_query} {' '.join(dict.fromkeys(retrieval_terms))}"

        source_constraint = None
        if re.search(r"\bpdf\b", normalized_for_intent):
            source_constraint = "pdf"
        elif re.search(r"\b(?:docx|resume)\b", normalized_for_intent):
            source_constraint = "docx"
        elif re.search(r"\bwhatsapp\b", normalized_for_intent):
            source_constraint = "whatsapp"

        topic_words = [word for word in keywords if not word.isdigit()]
        return {
            "intent": intent,
            "action": action,
            "topic": " ".join(topic_words[:8]),
            "entities": entities,
            "keywords": keywords,
            "question_number": question_number,
            "time_range": time_range if is_temporal else None,
            "source_constraint": source_constraint,
            "question_type": question_type,
            "retrieval_query": retrieval_query,
            "normalized_query": retrieval_query,
            "normalizations": normalizations,
            "entity_aliases": (["GATE", "DA", "DS", "Data Science", "Artificial Intelligence"]
                               if has_domain_entity else []),
        }

    @staticmethod
    def match_features(plan: dict, chunk: dict) -> dict:
        text = str(chunk.get("text") or "").casefold()
        source = str(chunk.get("source_name") or "").casefold()
        combined = f"{source} {text}"
        question_number = plan.get("question_number")
        question_match = False
        if question_number is not None:
            number = re.escape(str(question_number))
            question_match = bool(
                re.search(rf"\b(?:q(?:uestion)?\s*\.?\s*{number}|{number}\s*[.):])", text)
            )
        focus = [word.casefold() for word in plan.get("keywords", []) if len(word) > 1]
        aliases = [word.casefold() for word in plan.get("entity_aliases", [])]
        focus.extend(word for word in aliases if word not in focus)
        matched_terms = [word for word in focus if re.search(rf"\b{re.escape(word)}\b", combined)]
        source_matches = [word for word in focus if re.search(rf"\b{re.escape(word)}\b", source)]
        return {
            "question_number_match": question_match,
            "source_term_matches": len(source_matches),
            "topic_term_matches": len(matched_terms),
        }
