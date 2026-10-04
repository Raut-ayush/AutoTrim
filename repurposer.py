"""
AutoTrim - Content Repurposer

Takes a clean transcript.txt and generates:

    - X (Twitter) thread
    - Reddit post
    - Blog article

Uses a single Gemini API call with structured Pydantic output.

Usage:

    python repurposer.py "path/to/transcript.txt"

Example:

    python repurposer.py "output/Black Holes Explained – From Birth to Death/transcript.txt"

Optional:

    python repurposer.py "transcript.txt" --model gemini-3.8-flash
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ============================================================
# Configuration
# ============================================================

load_dotenv()

DEFAULT_MODEL = "gemini-3.8-flash"

# Rough estimate only.
# Actual tokenization depends on the model/tokenizer.
CHARS_PER_TOKEN_ESTIMATE = 4


# ============================================================
# Structured Output Schema
# ============================================================

class RepurposedContent(BaseModel):

    x_thread: list[str] = Field(
        description=(
            "A consecutive 4-6 tweet X/Twitter thread. "
            "Each tweet must be under 280 characters."
        )
    )

    reddit_title: str = Field(
        description=(
            "A neutral, analytical, thought-provoking "
            "Reddit post title."
        )
    )

    reddit_body: str = Field(
        description=(
            "The complete Reddit post body formatted "
            "in Markdown."
        )
    )

    blog_title: str = Field(
        description=(
            "An SEO-friendly, clear and compelling "
            "blog article title."
        )
    )

    blog_content: str = Field(
        description=(
            "The complete long-form blog article "
            "formatted in Markdown."
        )
    )


# ============================================================
# Prompts
# ============================================================

SYSTEM_PROMPT = """
You are an expert digital content strategist, science communicator,
technical writer, and storyteller.

Your job is to transform a raw video transcript into high-quality,
human-sounding content for X/Twitter, Reddit, and long-form blogs.

The final writing must feel like it was written by a thoughtful,
knowledgeable human — NOT like generic AI-generated content.

============================================================
GENERAL WRITING PRINCIPLES
============================================================

1. HUMAN WRITING

- Write naturally and conversationally.
- Vary sentence length and structure.
- Use specific observations instead of generic filler.
- Avoid robotic transitions such as:
  "In today's rapidly evolving world",
  "Let's dive in",
  "It's important to note",
  "In conclusion",
  unless genuinely appropriate.
- Avoid repetitive sentence patterns.
- Do not overuse em dashes.
- Don't make every paragraph follow the same structure.
- Prefer concrete language over corporate or academic jargon.
- Write with personality, but don't become overly casual.
- The reader should feel that a knowledgeable human actually wrote this.

2. ENGAGEMENT

- Create curiosity through ideas, discoveries, contradictions,
  surprising facts, or useful insights.
- Do NOT use cheap clickbait.
- Do NOT exaggerate claims just to increase engagement.
- Make the reader want to continue because the underlying idea
  is genuinely interesting.
- Use storytelling where it naturally fits the source material.
- Prefer interesting ideas over artificial "engagement hooks".

3. SOURCE FIDELITY

The transcript is the sole factual source.

- Do not invent facts, statistics, dates, examples, technical
  specifications, organizations, events, or claims that are not
  supported by the transcript.
- Do not add outside research.
- Do not silently correct statements from the transcript.
- Preserve uncertainty when the speaker expresses uncertainty.
- If the transcript says "we don't know", do not turn that into
  a definite scientific statement.
- If the transcript says "could", "may", "might", "we think",
  or similar language, preserve that uncertainty where relevant.
- You may reorganize, simplify, summarize, and rephrase the material,
  but preserve its original meaning.
- Never present speculation from the transcript as established fact.
- Do not strengthen a claim merely to make the writing more engaging.

4. NO MARKETING FLUFF

- Do not sound like an advertisement.
- Do not use exaggerated phrases such as:
  "This changes everything",
  "You won't believe",
  "Mind-blowing",
  "Game-changing",
  unless the source genuinely supports such language.
- Prioritize information and insight.
- Do not artificially manufacture excitement.

5. PLATFORM-NATIVE WRITING

Each output must feel specifically written for its platform.

Do NOT simply copy the same text three times and change the formatting.

============================================================
X / TWITTER THREAD
============================================================

- Create a high-signal 4-6 tweet thread.
- Start with an interesting hook based on an actual idea from
  the transcript.
- The hook should create curiosity without generic clickbait.
- Each tweet should communicate one clear idea.
- Maintain a logical progression from tweet to tweet.
- Use short paragraphs and natural spacing.
- Avoid unnecessary emojis.
- Avoid excessive hashtags.
- The final tweet should provide a meaningful conclusion or
  open-ended question when appropriate.
- Use 2-3 highly relevant hashtags only when they genuinely add value.
- Every tweet MUST be under 280 characters.
- Do not split a sentence awkwardly just to meet the character limit.
- Do not sacrifice factual accuracy to fit the character limit.

============================================================
REDDIT
============================================================

- Choose a title that is neutral, analytical, and interesting.
- Avoid sensationalism and clickbait.
- The post should feel like a genuine contribution to a community,
  not promotional content.
- Write a concise TL;DR or opening summary.
- Explain the most important facts and developments.
- Highlight interesting nuances, limitations, bottlenecks,
  contradictions, ethical questions, or trade-offs when they
  are actually present in the transcript.
- Use Markdown where useful.
- End with a genuine discussion question when appropriate.
- Do not use marketing language.
- Do not tell Reddit users to "like", "follow", "subscribe",
  "check out", or "share" the content.
- Do not mention that the content was generated from a transcript.

============================================================
BLOG
============================================================

- Create a clear, compelling, SEO-friendly title.
- Write a complete standalone article.
- Start with a strong hook or Executive Summary.
- Use Markdown H2 and H3 headings.
- Organize the article logically.
- Explain technical or complex concepts in accessible language.
- Include bullet points for important takeaways where useful.
- Maintain intellectual depth without becoming unnecessarily academic.
- Include a concluding perspective on the long-term implications.
- The article should feel like an edited human-written article,
  not an expanded transcript.
- Do not simply reproduce the transcript paragraph by paragraph.
- Do not invent information to make the article longer.
- Do not add external facts just to improve SEO.

============================================================
FINAL REQUIREMENT
============================================================

The three outputs should contain the same underlying facts but should
feel distinctly written for their respective audiences.

The goal is not simply to summarize the transcript.

The goal is to make the information genuinely useful, interesting,
readable, and natural on each platform while remaining faithful to
the source.
"""


USER_PROMPT_TEMPLATE = """
Analyze the following video transcript and create three distinct,
publication-ready pieces of content:

1. A 4-6 tweet X/Twitter thread
2. A Reddit post
3. A long-form blog article

Remember:

- Use ONLY information supported by the transcript.
- Make the writing human, natural, engaging, and intellectually interesting.
- Make it feel written by a knowledgeable human, not an AI.
- Do not sound like a marketing copywriter.
- Do not invent facts.
- Preserve uncertainty and nuance from the source.
- Make each piece native to its platform.
- The three outputs should not simply be copies of each other.

VIDEO TRANSCRIPT
================

{transcript}

================

Generate the structured output now.
"""


# ============================================================
# Gemini Client
# ============================================================

def create_client() -> genai.Client:
    """
    Create Gemini client using GEMINI_API_KEY from .env.
    """

    return genai.Client()


# ============================================================
# Validation
# ============================================================

def validate_content(
    data: RepurposedContent
) -> list[str]:
    """
    Validate important publishing constraints.

    Returns a list of warnings.
    """

    issues = []

    # --------------------------------------------------------
    # X thread count
    # --------------------------------------------------------

    if not 4 <= len(data.x_thread) <= 6:

        issues.append(
            f"X thread contains {len(data.x_thread)} tweets; "
            f"expected 4-6."
        )

    # --------------------------------------------------------
    # X tweet character count
    # --------------------------------------------------------

    for i, tweet in enumerate(data.x_thread, 1):

        length = len(tweet)

        if length > 280:

            issues.append(
                f"Tweet {i} is {length} characters (>280)."
            )

    # --------------------------------------------------------
    # Empty fields
    # --------------------------------------------------------

    if not data.reddit_title.strip():
        issues.append("Reddit title is empty.")

    if not data.reddit_body.strip():
        issues.append("Reddit body is empty.")

    if not data.blog_title.strip():
        issues.append("Blog title is empty.")

    if not data.blog_content.strip():
        issues.append("Blog content is empty.")

    return issues


# ============================================================
# Markdown Writers
# ============================================================

def save_x_thread(
    data: RepurposedContent,
    output_dir: Path
) -> Path:
    """
    Save X thread as Markdown.
    """

    path = output_dir / "x_thread.md"

    lines = []

    for i, tweet in enumerate(data.x_thread, 1):

        lines.append(
            f"**{i}/{len(data.x_thread)}**"
        )

        lines.append("")
        lines.append(tweet)
        lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    return path


def save_reddit_post(
    data: RepurposedContent,
    output_dir: Path
) -> Path:
    """
    Save Reddit post as Markdown.
    """

    path = output_dir / "reddit_post.md"

    content = (
        f"# {data.reddit_title}\n\n"
        f"{data.reddit_body}\n"
    )

    path.write_text(
        content,
        encoding="utf-8"
    )

    return path


def save_blog(
    data: RepurposedContent,
    output_dir: Path
) -> Path:
    """
    Save blog article as Markdown.
    """

    path = output_dir / "blog_post.md"

    content = (
        f"# {data.blog_title}\n\n"
        f"{data.blog_content}\n"
    )

    path.write_text(
        content,
        encoding="utf-8"
    )

    return path


def save_json(
    data: RepurposedContent,
    output_dir: Path
) -> Path:
    """
    Save structured result for future automation
    and publishing integrations.
    """

    path = output_dir / "repurposed.json"

    path.write_text(
        json.dumps(
            data.model_dump(),
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return path


# ============================================================
# Gemini Generation
# ============================================================

def generate_content(
    transcript: str,
    model: str,
    max_retries: int = 5
) -> RepurposedContent:
    """
    Generate all platform content using ONE Gemini request.
    """

    client = create_client()

    prompt = USER_PROMPT_TEMPLATE.format(
        transcript=transcript
    )

    last_error = None

    for attempt in range(1, max_retries + 1):

        try:

            print(
                "[Gemini] Generating content..."
            )

            print(
                f"[Gemini] Model: {model}"
            )

            start_time = time.time()

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=RepurposedContent,
                    temperature=0.3,
                ),
            )

            elapsed = time.time() - start_time

            print(
                f"[Gemini] Generation completed "
                f"in {elapsed:.2f}s"
            )

            # ------------------------------------------------
            # Preferred path:
            # google-genai parsed structured response
            # ------------------------------------------------

            if response.parsed is not None:

                return response.parsed

            # ------------------------------------------------
            # Fallback:
            # manually validate returned JSON
            # ------------------------------------------------

            if response.text:

                return RepurposedContent.model_validate_json(
                    response.text
                )

            raise RuntimeError(
                "Gemini returned an empty response."
            )

        except Exception as exc:

            last_error = exc

            print()
            print(
                f"[Gemini] Attempt "
                f"{attempt}/{max_retries} failed:"
            )

            print(
                f"         {exc}"
            )

            if attempt < max_retries:

                wait_time = 2 ** attempt

                print(
                    f"[Gemini] Retrying in "
                    f"{wait_time}s..."
                )

                time.sleep(wait_time)

    raise RuntimeError(
        "Gemini generation failed after "
        f"{max_retries} attempts: {last_error}"
    )


# ============================================================
# Main Repurposing Pipeline
# ============================================================

def repurpose_transcript(
    transcript_path: str,
    model: str = DEFAULT_MODEL
):
    """
    Read transcript.txt, generate content and save
    all platform outputs.
    """

    transcript_file = Path(transcript_path)

    # --------------------------------------------------------
    # Check transcript
    # --------------------------------------------------------

    if not transcript_file.exists():

        raise FileNotFoundError(
            f"Transcript not found:\n"
            f"{transcript_file}"
        )

    if transcript_file.suffix.lower() != ".txt":

        print(
            "Warning: expected a .txt transcript, "
            f"got {transcript_file.suffix}"
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 60)
    print("AutoTrim - Content Repurposer")
    print("=" * 60)

    print(
        f"Transcript: {transcript_file}"
    )

    print(
        f"Model:      {model}"
    )

    print()

    # --------------------------------------------------------
    # Read transcript
    # --------------------------------------------------------

    transcript = transcript_file.read_text(
        encoding="utf-8"
    ).strip()

    if not transcript:

        raise ValueError(
            "Transcript is empty."
        )

    # --------------------------------------------------------
    # Basic input statistics
    # --------------------------------------------------------

    character_count = len(transcript)

    estimated_tokens = (
        character_count /
        CHARS_PER_TOKEN_ESTIMATE
    )

    print(
        f"Transcript length: "
        f"{character_count:,} characters"
    )

    print(
        f"Estimated input tokens: "
        f"~{estimated_tokens:,.0f}"
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    data = generate_content(
        transcript=transcript,
        model=model
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    issues = validate_content(data)

    print()

    if issues:

        print(
            "[Validation] Warnings:"
        )

        for issue in issues:

            print(
                f"  - {issue}"
            )

    else:

        print(
            "[Validation] All checks passed."
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        transcript_file.parent /
        "content"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    x_path = save_x_thread(
        data,
        output_dir
    )

    reddit_path = save_reddit_post(
        data,
        output_dir
    )

    blog_path = save_blog(
        data,
        output_dir
    )

    json_path = save_json(
        data,
        output_dir
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Content generation complete")
    print("=" * 60)

    print(
        f"X thread:   {x_path}"
    )

    print(
        f"Reddit:     {reddit_path}"
    )

    print(
        f"Blog:       {blog_path}"
    )

    print(
        f"JSON:       {json_path}"
    )

    print()

    print(
        f"X tweets:   {len(data.x_thread)}"
    )

    print(
        f"Reddit chars: "
        f"{len(data.reddit_body):,}"
    )

    print(
        f"Blog chars:   "
        f"{len(data.blog_content):,}"
    )

    return data


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate X, Reddit and Blog content "
            "from a transcript."
        )
    )

    parser.add_argument(
        "transcript",
        help="Path to transcript.txt"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Gemini model "
            f"(default: {DEFAULT_MODEL})"
        )
    )

    args = parser.parse_args()

    try:

        repurpose_transcript(
            transcript_path=args.transcript,
            model=args.model
        )

    except KeyboardInterrupt:

        print(
            "\nCancelled by user."
        )

    except Exception as exc:

        print()
        print(
            f"ERROR: {exc}"
        )

        raise SystemExit(1)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()