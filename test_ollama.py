from llm.ollama_provider import OllamaProvider


MODEL = "qwen2.5-coder:7b"


def main():
    ollama = OllamaProvider(model=MODEL)

    print("=" * 60)
    print("AutoTrim - Ollama Test")
    print("=" * 60)

    print(f"Model: {MODEL}")

    if not ollama.check_connection():
        print("\nERROR: Ollama is not running.")
        print("Start Ollama and try again.")
        return

    print("Ollama: Connected")
    print("\nGenerating...\n")

    result = ollama.generate(
        prompt=(
            "Explain quantum superposition in exactly "
            "three simple sentences."
        ),
        system=(
            "You are a content processing assistant. "
            "Be concise and factual."
        ),
    )

    print(result["response"])

    print("\n" + "-" * 60)
    print(f"Generation time: {result['total_time']:.2f}s")
    print(f"Prompt tokens: {result['prompt_tokens']}")
    print(f"Output tokens: {result['output_tokens']}")
    print("Model unload requested: yes")
    print("-" * 60)


if __name__ == "__main__":
    main()