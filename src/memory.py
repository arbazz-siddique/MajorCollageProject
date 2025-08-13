# Add a new question and answer pair to memory and keep only the most recent entries
def add_to_memory(memory, question, answer, max_memory=3):
    """Improved memory with deduplication"""
    # Remove exact duplicates
    memory = [(q, a) for q, a in memory if q != question]
    memory.append((question, answer))
    return memory[-max_memory:]

# Format the memory into a readable string block for prompt injection
def format_memory_prompt(memory):
    if not memory:
        return ""
    formatted = "\n\n".join(
        [f"Previous Q: {q}\nPrevious A: {a}" for q, a in memory]
    )
    return f"\n\n[CONTEXT FROM PAST INTERACTIONS]\n{formatted}"