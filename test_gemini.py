from llm.gemini_provider import GeminiProvider


def main():

    print("=" * 60)
    print("AutoTrim - Gemini Test")
    print("=" * 60)

    try:
        gemini = GeminiProvider()

        print("Model    :", gemini.model)
        print("API      : Connected")

    except Exception as e:
        print("ERROR:", e)
        return

    print()
    print("Generating...")

    try:

        result = gemini.generate(
            prompt=(
                "Explain quantum superposition in exactly "
                "3 simple sentences."
            ),
            system=(
                "You are a helpful scientific content assistant. "
                "Be accurate and concise."
            ),
            temperature=0.2,
        )

    except Exception as e:

        print()
        print("Gemini generation failed.")
        print(e)
        return

    print()
    print("Response:")
    print(result["response"])

    print()
    print(f"Generation time: {result['total_time']:.2f}s")


if __name__ == "__main__":
    main()