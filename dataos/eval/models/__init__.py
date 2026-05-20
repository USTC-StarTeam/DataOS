"""DataOS-provided lm-eval model backends."""

# Import side effects register the model aliases with lm-eval.
from dataos.eval.models.openai_tool_chat import DataOSLocalToolChatCompletion  # noqa: F401
