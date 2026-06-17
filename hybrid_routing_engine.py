import re
import json
import time


class HybridSentimentEngine:
    """Two-tier sentiment analysis: fast regex sieve → heavy LLM fallback."""

    def __init__(self, llama_model):
        """
        Initializes the Hybrid Engine.
        Takes the already-loaded llama-cpp Llama model as an argument.
        """
        self.llama = llama_model

        # Macro Paradox Triggers
        self.paradox_triggers = [
            r"inflation", r"fed\b", r"yields?\s*spike", r"rate[sd]?\s*(hike|cut|rise)",
            r"cpi\b", r"guidance\s*(cut|lower)", r"diluti(?:on|ve)",
            r"lawsuit", r"fined?", r"tariff", r"sanctions?",
            r"recession", r"default", r"geopolitical",
            r"but\s+(the\s+)?stock\s+(fell|dropped|slid)",
            r"despite", r"however",
        ]

        # Fast-Path Bullish Keywords
        self.bullish_keywords = [
            r"beats?\s*estimates", r"revenue\s+up", r"record\s+profit",
            r"raised\s+guidance", r"upgraded", r"all[- ]time\s+high",
            r"strong\s+earnings", r"dividend\s+(hike|increase)",
            r"expand(?:s|ing|ed)?", r"growth", r"opening"
        ]

        # Fast-Path Bearish Keywords
        self.bearish_keywords = [
            r"missed?\s*estimates", r"revenue\s+down", r"bankruptcy",
            r"downgrade[ds]?", r"layoffs?", r"plunged?",
            r"warned?\s+of\s+losses", r"profit\s+warning",
        ]

    def _fast_path_scan(self, headline: str) -> dict | None:
        """Regex keyword scan. Returns None if too complex for fast path."""
        headline_lower = headline.lower()

        # RULE 1: Check for Paradoxes. If found, ABORT FAST PATH.
        for trigger in self.paradox_triggers:
            if re.search(trigger, headline_lower):
                print(f"[ROUTER] ⚡ Paradox trigger '{trigger}' detected → Routing to Heavy LLM...")
                return None

        # RULE 2: Check for obvious Bullish signals
        for keyword in self.bullish_keywords:
            if re.search(keyword, headline_lower):
                return {
                    "label": "positive",
                    "reasoning": f"Fast-path keyword match: {keyword.replace(chr(92), '')}",
                    "confidence": {"positive": 85, "negative": 5, "neutral": 10},
                    "engine_used": "⚡ Fast-Path Algorithmic",
                }

        # RULE 3: Check for obvious Bearish signals
        for keyword in self.bearish_keywords:
            if re.search(keyword, headline_lower):
                return {
                    "label": "negative",
                    "reasoning": f"Fast-path keyword match: {keyword.replace(chr(92), '')}",
                    "confidence": {"positive": 5, "negative": 85, "neutral": 10},
                    "engine_used": "⚡ Fast-Path Algorithmic",
                }

        # If nothing matches, DO NOT return Neutral.
        # This forces the router to send the headline to Llama for deeper analysis.
        return None

    def _slow_path_llama(self, headline: str) -> dict:
        """Deep LLM inference for ambiguous / paradoxical headlines."""
        print("[ROUTER] 🧠 Executing Deep Llama-3 Analysis...")

        system_prompt = (
            "You are a Quantitative Finance Sentiment Engine. "
            "YOUR RULES:\n"
            "1. IF 'guidance', 'outlook', or 'forecast' is lowered or cut: NEGATIVE.\n"
            "2. IF 'expand', 'growth', or 'opening' is mentioned regarding operations: POSITIVE.\n"
            "3. IF 'profit fell' or 'misses': NEGATIVE.\n"
            "4. IF none of the above, focus on the fundamental market trajectory.\n"
            "5. Output ONLY a valid JSON object. No other text.\n"
            "Format: {'sentiment': 'POSITIVE'|'NEGATIVE'|'NEUTRAL', 'reasoning_token_focus': '...'}"
        )

        formatted_prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\nHeadline: {headline}\n\n"
            f"Step 1: Check for guidance cuts.\n"
            f"Step 2: Determine sentiment based on rules.\n"
            f"Step 3: Output JSON.<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{{"
        )

        response = self.llama.create_completion(
            prompt=formatted_prompt,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            stop=["<|eot_id|>"],
        )

        raw_output = "{" + response["choices"][0]["text"].strip()
        print(f"\n--- RAW MODEL OUTPUT ---\n{raw_output}\n------------------------\n")

        # JSON Parsing (brace-counting fallback)
        sentiment = "neutral"
        reasoning = "No reasoning provided."
        data = {}

        def extract_json(text):
            """Extract the outermost JSON object by counting braces."""
            start_idx = text.find("{")
            if start_idx == -1:
                return None
            depth = 0
            for i in range(start_idx, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start_idx : i + 1]
            return None

        try:
            data = json.loads(raw_output)
            sentiment = data.get("sentiment", "neutral").lower()
            reasoning = data.get("reasoning_token_focus", reasoning)
        except json.JSONDecodeError:
            json_str = extract_json(raw_output)
            if json_str:
                try:
                    data = json.loads(json_str)
                    sentiment = data.get("sentiment", "neutral").lower()
                    # Grab conversational reasoning outside the JSON block
                    full_text = raw_output.replace(json_str, "").strip()
                    full_text = re.sub(
                        r"^(Here is the analysis:|Reasoning:)\s*",
                        "", full_text, flags=re.IGNORECASE,
                    ).strip()
                    reasoning = full_text if full_text else data.get("reasoning_token_focus", reasoning)
                except json.JSONDecodeError:
                    reasoning = f"Parse failed. Raw: {raw_output}"
            else:
                lower_out = raw_output.lower()
                if "negative" in lower_out:
                    sentiment = "negative"
                elif "positive" in lower_out:
                    sentiment = "positive"
                reasoning = f"Extracted from raw output: {raw_output[:200]}"

        # Build confidence scores
        confidence = {"positive": 0, "negative": 0, "neutral": 0}
        try:
            conf_data = data.get("confidence_scores", {})
            confidence["positive"] = int(conf_data.get("positive", 0))
            confidence["negative"] = int(conf_data.get("negative", 0))
            confidence["neutral"] = int(conf_data.get("neutral", 0))
        except Exception:
            pass

        if sum(confidence.values()) == 0:
            if sentiment == "positive":
                confidence = {"positive": 78, "negative": 8, "neutral": 14}
            elif sentiment == "negative":
                confidence = {"positive": 7, "negative": 80, "neutral": 13}
            else:
                confidence = {"positive": 20, "negative": 15, "neutral": 65}

        return {
            "label": sentiment,
            "reasoning": reasoning,
            "confidence": confidence,
            "engine_used": "🧠 Llama-3-8B Heavy LLM",
        }

    def analyze(self, headline: str) -> dict:
        """Route headline through fast path; fall back to heavy LLM if needed."""
        start_time = time.perf_counter()

        # 1. Attempt Fast Path
        result = self._fast_path_scan(headline)

        # 2. If Fast Path bailed (returned None), use heavy Llama model
        if result is None:
            result = self._slow_path_llama(headline)

        latency = (time.perf_counter() - start_time) * 1000
        result["latency_ms"] = round(latency, 2)

        return result
