from src.memory import add_to_memory, format_memory_prompt

def test_memory_trims_correctly():
    # Simulates adding more items than the memory limit allows
    mem = []
    mem = add_to_memory(mem, "Q1", "A1", max_memory=2)
    mem = add_to_memory(mem, "Q2", "A2", max_memory=2)
    mem = add_to_memory(mem, "Q3", "A3", max_memory=2)
    assert len(mem) == 2
    assert mem[0][0] == "Q2"
    assert mem[1][0] == "Q3"

def test_memory_formatting():
    # Formats a single Q&A pair into a context block
    mem = [("What is AI?", "It is the science of making machines smart.")]
    result = format_memory_prompt(mem)
    assert "Previous Q" in result
    assert "Previous A" in result