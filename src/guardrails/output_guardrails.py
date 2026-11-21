"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.config import DEMO_SECRETS
from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

# Thứ tự quan trọng: secret/password trước (khớp dài nhất), rồi tới email,
# rồi các dãy số (phone trước national_id để 10–11 số không bị đếm nhầm).
PII_PATTERNS: dict[str, str] = {
    # Secret nguyên văn từ data/protected/vinbank_secrets.json
    "secret_value": "|".join(
        re.escape(s) for s in sorted(DEMO_SECRETS, key=len, reverse=True)
    ) or r"(?!x)x",
    # Host nội bộ: db.vinbank.internal:5432, cache.corp.local …
    "internal_host": r"\b[\w-]+(?:\.[\w-]+)*\.(?:internal|local|corp|lan)(?::\d{2,5})?\b",
    # API key kiểu sk-..., sk_live_..., Bearer token
    "api_key": r"\bsk[-_][A-Za-z0-9_-]{6,}|\bBearer\s+[A-Za-z0-9._-]{16,}",
    # password=..., password is ..., mật khẩu là ..., pwd: ...
    "password": (
        r"(?:password|passwd|pwd|pass\s*code|mật\s*khẩu|mat\s*khau)"
        r"\s*(?:is|là|la|[:=])\s*\S+"
    ),
    "email": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}",
    # SĐT VN: di động 10 số / bàn 11 số, dạng 0xxx hoặc +84xxx.
    # Ranh giới: không dính chữ số liền kề và không nằm trong số thập phân/có
    # phân cách (1.000.000) — nhưng dấu chấm cuối câu thì vẫn bắt.
    "phone": r"(?<!\d)(?<!\d[.,])(?:\+84|0)\d{9,10}(?!\d)(?![.,]\d)",
    # CCCD 12 số / CMND 9 số — bỏ qua khi là số tiền (…000 VND / đ)
    "national_id": (
        r"(?<!\d)(?<!\d[.,])(?:\d{12}|\d{9})(?!\d)(?![.,]\d)"
        r"(?!\s*(?:vnd|vnđ|đ|dong|đồng))"
    ),
}
_COMPILED_PII = {k: re.compile(v, re.IGNORECASE) for k, v in PII_PATTERNS.items()}

# Secret bị làm rối (chèn space/dấu chấm, viết hoa…) → so trên bản chỉ còn chữ+số.
_SECRET_NEEDLES = sorted(
    {re.sub(r"[^a-z0-9]", "", s.lower()) for s in DEMO_SECRETS if s}, key=len
)

SAFE_FALLBACK_MESSAGE = (
    "I'm sorry, I can't share that information. "
    "How else can I help with your VinBank account or banking needs?"
)


def _obfuscated_secret_present(text: str) -> bool:
    squashed = re.sub(r"[^a-z0-9]", "", (text or "").lower())
    return any(n and n in squashed for n in _SECRET_NEEDLES)


def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
        (+ 'issue_types': list tên loại vi phạm, 'withheld': True nếu phải thay
        toàn bộ câu trả lời vì secret bị làm rối không redact được từng đoạn).
    """
    response = response or ""
    issues: list[str] = []
    issue_types: list[str] = []
    redacted = response

    for name, rx in _COMPILED_PII.items():
        # Phát hiện trên bản gốc (để báo đủ loại: secret_value + api_key),
        # thay thế trên bản đang redact dần.
        matches = rx.findall(response)
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            issue_types.append(name)
            redacted = rx.sub("[REDACTED]", redacted)

    # Lớp cuối: secret vẫn còn sau khi redact (bị tách ký tự, đổi hoa thường…)
    withheld = False
    if _obfuscated_secret_present(redacted):
        issues.append("obfuscated_secret: protected value still present after redaction")
        issue_types.append("obfuscated_secret")
        redacted = SAFE_FALLBACK_MESSAGE
        withheld = True

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "issue_types": issue_types,
        "redacted": redacted,
        "withheld": withheld,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        self.last_decision: str | None = None  # cho audit log

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        self.last_decision = None

        # 1. Regex filter — deterministic, luôn chạy.
        result = content_filter(response_text)
        if not result["safe"]:
            self.redacted_count += 1
            if result.get("withheld"):
                self.blocked_count += 1
            self.last_decision = "redacted:" + ",".join(result["issue_types"])
            response_text = result["redacted"]
            llm_response.content = self._make_content(response_text)

        # 2. LLM-as-Judge (optional) — chạy trên bản đã redact.
        if self.use_llm_judge:
            verdict = await llm_safety_check(response_text)
            if not verdict["safe"]:
                self.blocked_count += 1
                self.last_decision = "judge_unsafe"
                llm_response.content = self._make_content(SAFE_FALLBACK_MESSAGE)

        return llm_response

    @staticmethod
    def _make_content(text: str) -> types.Content:
        return types.Content(role="model", parts=[types.Part.from_text(text=text)])


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")

    print("\nLab dataset (data/pii_hallucination_samples.json → pii_cases):")
    passed = 0
    cases = load_lab_pii_dataset().get("pii_cases", [])
    for case in cases:
        result = content_filter(case["input_text"])
        ok = result["safe"] == case["expect_safe"] and set(
            case.get("expect_issue_types", [])
        ) <= set(result["issue_types"])
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {case['id']:<8} {result['issue_types']}")
    print(f"  → {passed}/{len(cases)} PII cases match expectations")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
