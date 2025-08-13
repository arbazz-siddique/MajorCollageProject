from src.generator import build_prompt

# Reusable dummy docs for prompt construction
def dummy_docs():
    return [
        ("Deep learning is a subset of ML.", {"title": "Intro to Deep Learning", "url": "http://test.com"}),
        ("Transformers revolutionized NLP.", {"title": "Transformers Paper", "url": "http://test2.com"})
    ]

# Test prompt generation with default settings
def test_prompt_default():
    prompt = build_prompt("What is deep learning?", dummy_docs())
    assert "What is deep learning?" in prompt
    assert "Context" in prompt
    assert "Intro to Deep Learning" in prompt
    assert "Transformers Paper" in prompt

# Test that CoT instruction is added when cot=True
def test_prompt_with_cot():
    prompt = build_prompt("Explain RL", dummy_docs(), cot=True)
    assert "step by step" in prompt.lower()

# Test that "Concise" style instruction appears in prompt
def test_prompt_concise_style():
    prompt = build_prompt("Summarize CNNs", dummy_docs(), style="Concise")
    assert "briefly and directly" in prompt.lower()

# Test that "Beginner-Friendly" style instruction appears in prompt
def test_prompt_beginner_friendly_style():
    prompt = build_prompt("What is backpropagation?", dummy_docs(), style="Beginner-Friendly")
    assert "as if to someone new to ai" in prompt.lower()

# Test that "Explain Step-by-Step" style includes reasoning flow language
def test_prompt_explain_step_by_step_style():
    prompt = build_prompt("How does Q-learning work?", dummy_docs(), style="Explain Step-by-Step")
    assert "explaining your reasoning step by step" in prompt.lower()

# Test that "With Citations Only" restricts to source-backed answers
def test_prompt_with_citations_only_style():
    prompt = build_prompt("What is gradient descent?", dummy_docs(), style="With Citations Only")
    assert "only the information provided in the sources" in prompt.lower()