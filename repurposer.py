"""
AutoTrim - Content Repurposer

Takes a clean transcript.txt and generates:

    - X (Twitter) thread
    - Reddit post
    - Substack post

Uses a single Gemini API call with structured Pydantic output.

Usage:

    python repurposer.py "path/to/transcript.txt"

Example:

    python repurposer.py "output/Black Holes Explained/transcript.txt"

Optional:

    python repurposer.py "transcript.txt" --model gemini-3.8-flash
"""

import argparse
import json
import re
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

# X safety limit.
# We intentionally use 270 instead of 280 to leave some margin.
MAX_TWEET_CHARS = 270


# ============================================================
# Structured Output Schema
# ============================================================

class RepurposedContent(BaseModel):

    x_thread: list[str] = Field(
        description=(
            "A consecutive 4-6 tweet X/Twitter thread. "
            "Each tweet should be 270 characters or fewer."
        )
    )

    reddit_title: str = Field(
        description=(
            "A neutral, specific and interesting Reddit post title. "
            "It should sound like a genuine community post, not clickbait."
        )
    )

    reddit_body: str = Field(
        description=(
            "The complete Reddit post body formatted in Markdown. "
            "It should be conversational, analytical and community-oriented."
        )
    )

    substack_title: str = Field(
        description=(
            "A clear, compelling and natural Substack title. "
            "It should create curiosity without clickbait."
        )
    )

    substack_content: str = Field(
        description=(
            "The complete long-form Substack post formatted in Markdown. "
            "It should read like a thoughtful human-written newsletter or essay."
        )
    )


# ============================================================
# Prompts
# ============================================================

SYSTEM_PROMPT = """
You are an expert digital content strategist, science communicator,
technical writer, and editor.

Your job is to transform a video transcript into three distinct,
publication-ready pieces of content:

1. X/Twitter thread
2. Reddit post
3. Substack post

The final writing must feel like it was written by a thoughtful,
knowledgeable human, not generic AI-generated content.

The transcript is the source of truth.

============================================================
RULE 1: SOURCE FIDELITY
============================================================

The transcript is the ONLY factual source.

You may reorganize, simplify, summarize and rephrase what is said,
but you must not introduce information that is not supported by the
transcript.

Follow these rules:

- Do not invent facts, statistics, dates, examples, technical
  specifications, organizations, events or claims.
- Do not add outside research or background knowledge.
- Do not silently correct statements from the transcript.
- Do not turn an uncertain statement into a certain one.
- If the speaker says "could", "may", "might", "probably",
  "we think", "we don't know", or similar language, preserve
  the uncertainty where relevant.
- Never strengthen a claim merely to make it more interesting.
- Do not invent cause-and-effect relationships.
- Do not invent conclusions, predictions or implications.
- Do not add opinions that are not present in the transcript.
- Keep important numbers, names, units and technical terms faithful
  to the transcript.
- If the transcript leaves something unexplained, leave it unexplained.
  Do not fill the gap using outside knowledge.
- If the transcript contains speculation, present it as speculation.
- If the transcript contains an apparent factual error, do not silently
  fix it.

The goal is NOT to reproduce the transcript sentence by sentence.

The goal is to extract the ideas from the transcript and express them
naturally for each platform without adding unsupported information.


============================================================
RULE 2: TRANSCRIPT CLEANUP
============================================================

The transcript was produced using automatic speech recognition.

It may contain:

- misheard words
- incorrect punctuation
- repeated phrases
- filler words
- false starts
- unfinished sentences
- minor spelling errors

Clean these up silently when the intended meaning is clear.

If a word or technical term appears garbled and you cannot determine
what it means confidently:

- do not guess
- omit the uncertain detail or describe it more generally

Do not use the speaker's name unless it is clearly present in the
transcript.

Ignore obvious sponsor messages, calls to subscribe, like/share
requests and channel housekeeping unless they are genuinely part of
the subject being discussed.


============================================================
RULE 3: HUMAN WRITING
============================================================

Write like a knowledgeable human editor.

DO:

- Use plain, direct language.
- Prefer concrete wording.
- Vary sentence length.
- Vary paragraph length.
- Avoid repetitive structures.
- Start with the actual idea instead of generic introductions.
- Let interesting ideas create engagement naturally.
- Use the source's concrete examples when useful.
- Reorganize information when doing so improves readability without
  changing meaning.
- Keep the tone confident when the source is confident.
- Keep the tone uncertain when the source is uncertain.

DO NOT:

- Sound like a marketing copywriter.
- Use generic AI introductions.
- Pad the writing.
- Repeat the same idea in different words.
- Make every paragraph follow the same structure.
- Make every sentence sound dramatic.
- Manufacture a "big takeaway".
- Manufacture emotional reactions.
- Use unnecessary rhetorical questions.
- Use fake personal experiences.
- Write as though you personally watched, tested or researched
  something.
- Use "I" unless the source itself requires a first-person quotation
  or statement.
- Use emojis.
- Use em dashes.
- Use excessive exclamation marks.
- Use clickbait or exaggerated language.

Avoid stock phrases such as:

- "In today's rapidly evolving world"
- "Let's dive in"
- "Let's explore"
- "It's important to note"
- "It's worth noting"
- "In conclusion"
- "At the end of the day"
- "game-changing"
- "mind-blowing"
- "unlock"
- "unleash"
- "buckle up"
- "fascinating world"
- "ever-evolving landscape"

Do not repeatedly use the pattern:

"It's not just X, it's Y."

Do not use that structure as a writing habit.


============================================================
RULE 4: ENGAGEMENT
============================================================

Make the content interesting because the underlying ideas are
interesting.

Good engagement comes from:

- an unusual idea
- a surprising fact actually present in the transcript
- a contradiction
- an interesting technical detail
- a useful explanation
- an unresolved question actually discussed
- a meaningful consequence explicitly supported by the source

Do NOT create engagement by:

- exaggerating
- withholding information artificially
- creating fake suspense
- inventing stakes
- using clickbait
- adding unsupported implications

The reader should continue because the content itself is worth reading.


============================================================
FORMAT 1: X / TWITTER THREAD
============================================================

Create 4-6 tweets.

Every tweet must be 270 characters or fewer.

Tweet 1:

- Should contain the most interesting real idea from the transcript.
- Should work as a natural hook.
- Must not tease something the thread does not actually explain.
- Do not use generic clickbait.

Tweets 2 onward:

- Each should communicate one clear idea.
- Maintain a logical progression.
- Do not make every tweet sound like a separate viral hook.
- Do not repeat the same point.
- Each tweet should be understandable on its own.
- Never split a sentence awkwardly across tweets.
- Keep the writing concise.

The final tweet:

- Should contain the final meaningful point from the source.
- A question is allowed only when naturally supported by the content.
- Do not manufacture a discussion question.

Hashtags:

- None are acceptable.
- At most 2 hashtags.
- Only use them when clearly relevant.
- If used, place them in the final tweet.

Do not use:

- "1/"
- "1/6"
- thread numbering

Numbering is added automatically when the Markdown file is created.

Do not include:

- "Follow for more"
- "Like"
- "Retweet"
- "Share"
- promotional calls to action


============================================================
FORMAT 2: REDDIT POST
============================================================

TITLE:

- Neutral.
- Specific.
- Informative.
- Interesting without being sensational.
- Under 150 characters.
- Should sound like a real Reddit user wrote it.

BODY:

Start with:

TL;DR:

The TL;DR should summarize the central idea in 1-2 sentences.

Then:

- Explain the important points in short, readable paragraphs.
- Use Markdown where genuinely useful.
- Use bullets only when they improve readability.
- Include relevant nuances, limitations, caveats or unresolved points
  that are actually present in the transcript.
- Do not turn the post into a formal article.
- Write conversationally but intelligently.
- It should feel like a knowledgeable Reddit community contribution.

You may end with a discussion question, but only when the transcript
naturally supports one.

Do not:

- sound like a journalist writing a news article
- sound like marketing
- mention that the content came from a transcript
- mention AI or automatic generation
- tell readers to follow, subscribe, like, share or check something out
- invent a personal opinion
- write "I think" or "I believe" unless it is directly supported by
  the source


============================================================
FORMAT 3: SUBSTACK POST
============================================================

The Substack output is NOT a generic SEO blog.

It should feel like a thoughtful newsletter or essay written by a
knowledgeable human.

TITLE:

- Clear.
- Specific.
- Interesting.
- Natural.
- Curiosity-driven without clickbait.
- Avoid formulaic titles.
- Avoid excessive use of colons.
- Do not simply copy the video title.

OPENING:

Start directly with the most interesting idea supported by the
transcript.

Do not begin with:

- "In this article..."
- "In today's world..."
- "Let's explore..."
- generic scene-setting
- a description of what the reader is about to read

The opening should make the reader understand why the subject
is interesting based entirely on the source.

BODY:

- Write a coherent standalone piece.
- Use Markdown H2 headings when they genuinely improve navigation.
- H3 headings should be used sparingly.
- Do not add headings simply to make the article look structured.
- Explain technical concepts clearly using only information supported
  by the transcript.
- Preserve useful examples and analogies from the speaker.
- Reorganize information when it improves the reading experience.
- Connect related ideas naturally.
- Use paragraphs rather than turning everything into bullet points.
- Bullet lists are allowed when they genuinely improve readability.
- Do not write sentence-by-sentence from the transcript.
- Do not simply expand the transcript.

The article should feel like something a reader would willingly
subscribe to a newsletter to receive.

ENDING:

Do not force a conclusion.

If the source naturally reaches a conclusion, preserve it.

If the source ends with an open question, uncertainty or unresolved
idea, it is acceptable to end there.

Do not add:

- a generic "Conclusion" section
- a motivational message
- predictions
- "what this means for the future"
- broader implications
- a call to action

unless those ideas are actually supported by the transcript.

Do not mention:

- the transcript
- AI
- automatic generation
- the original video
- the content creation process

unless the transcript itself makes that information relevant.


============================================================
RULE 5: DIFFERENT PLATFORMS, SAME FACTS
============================================================

The three outputs must contain the same underlying factual information,
but they must NOT feel like copies.

X should be:

- concise
- high-signal
- sequential
- easy to scan

Reddit should be:

- conversational
- analytical
- community-oriented
- discussion-friendly

Substack should be:

- thoughtful
- explanatory
- narrative
- comfortable to read at length

Do not simply shorten the Substack post into Reddit and then shorten
Reddit into X.

Each piece must be independently written for its platform.


============================================================
RULE 6: QUOTATIONS
============================================================

Do not put text in quotation marks unless it is an exact short phrase
from the transcript.

Do not use quotation marks merely to make a phrase sound authoritative.

Do not create fake quotations.


============================================================
RULE 7: LENGTH
============================================================

Length should follow the amount of useful information in the transcript.

These are targets, not requirements.

Under 300 words:

- Reddit: roughly 100-200 words
- Substack: roughly 250-400 words

300-2,000 words:

- Reddit: roughly 150-350 words
- Substack: roughly 500-1,000 words

2,000-8,000 words:

- Reddit: roughly 250-500 words
- Substack: roughly 900-1,800 words

Over 8,000 words:

- Reddit: roughly 350-600 words
- Substack: roughly 1,500-2,500 words

If the transcript is thin, write less.

Never add information merely to reach a target length.

X is always 4-6 tweets of 270 characters or fewer.


============================================================
RULE 8: DO NOT ADD SOURCE CREDITS
============================================================

Do not add:

- source credit
- attribution
- "via"
- links
- channel name
- video title as a source section
- bibliography
- references

Credits and links will be handled separately by AutoTrim.


============================================================
FINAL SELF-CHECK
============================================================

Before returning the structured response, silently check:

1. Does every factual statement come from the transcript?
2. Did you accidentally add outside knowledge?
3. Did you preserve uncertainty?
4. Did you change the meaning of anything?
5. Did you invent cause and effect?
6. Did you invent a conclusion or implication?
7. Did you invent an example?
8. Did you create a quotation that is not exact?
9. Does every X tweet stay within 270 characters?
10. Are there 4-6 X tweets?
11. Is the Reddit title under 150 characters?
12. Does Reddit begin with TL;DR?
13. Does Substack read like a newsletter rather than an SEO article?
14. Are the three outputs genuinely different?
15. Did you avoid generic AI language?
16. Did you avoid marketing language?
17. Did you avoid padding thin source material?

If any answer is no, fix the output before returning it.

Return only the requested structured fields.
"""


USER_PROMPT_TEMPLATE = """
Analyze the following video transcript and create three distinct,
publication-ready pieces of content:

1. A 4-6 tweet X/Twitter thread
2. A Reddit post with title and body
3. A long-form Substack post with title and body

Important:

- Use ONLY information supported by the transcript.
- Do not add outside knowledge or research.
- Preserve uncertainty and nuance.
- Do not invent facts, examples, implications or conclusions.
- Write naturally and like a knowledgeable human.
- Make each piece native to its platform.
- Do not simply copy one piece into another.
- Do not pad the content.

For Substack specifically:

- Write a thoughtful newsletter/essay, not an SEO blog.
- Start with the most interesting idea.
- Use a natural narrative flow.
- Use headings only when useful.
- Do not force a conclusion.
- Do not add information merely to make the article more complete.

For X specifically:

- 4-6 tweets.
- Maximum 270 characters per tweet.
- First tweet should be the strongest supported hook.
- Do not make every tweet sound like a viral hook.
- No numbering.
- At most 2 relevant hashtags.

For Reddit specifically:

- Neutral, specific title.
- Begin the body with "TL;DR:".
- Write like a knowledgeable community member.
- Include useful nuance from the transcript.
- Avoid promotional language.

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
# Error Classification
# ============================================================

def is_non_retryable_error(exc: Exception) -> bool:
    """
    Determine whether an error should stop immediately instead
    of being retried.

    In particular, daily quota exhaustion should not trigger
    five pointless retries.
    """

    error_text = str(exc).lower()

    non_retryable_patterns = [
        "quota exceeded",
        "generaterequestsperdayperprojectpermodel-freetier",
        "resource_exhausted",
        "exceeded your current quota",
        "daily quota",
        "billing details",
        "api key not valid",
        "invalid api key",
        "permission denied",
        "unauthenticated",
    ]

    return any(
        pattern in error_text
        for pattern in non_retryable_patterns
    )


# ============================================================
# Validation
# ============================================================

def validate_content(
    data: RepurposedContent
) -> list[str]:
    """
    Validate important publishing constraints.

    Returns a list of warnings/errors.
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

        if length > MAX_TWEET_CHARS:
            issues.append(
                f"Tweet {i} is {length} characters "
                f"(>{MAX_TWEET_CHARS})."
            )

        if not tweet.strip():
            issues.append(
                f"Tweet {i} is empty."
            )

    # --------------------------------------------------------
    # Reddit
    # --------------------------------------------------------

    if not data.reddit_title.strip():
        issues.append(
            "Reddit title is empty."
        )

    if len(data.reddit_title) > 150:
        issues.append(
            f"Reddit title is {len(data.reddit_title)} "
            "characters (>150)."
        )

    if not data.reddit_body.strip():
        issues.append(
            "Reddit body is empty."
        )

    # --------------------------------------------------------
    # Substack
    # --------------------------------------------------------

    if not data.substack_title.strip():
        issues.append(
            "Substack title is empty."
        )

    if not data.substack_content.strip():
        issues.append(
            "Substack content is empty."
        )

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


def save_substack_post(
    data: RepurposedContent,
    output_dir: Path
) -> Path:
    """
    Save Substack post as Markdown.
    """

    path = output_dir / "substack_post.md"

    content = (
        f"# {data.substack_title}\n\n"
        f"{data.substack_content}\n"
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

            # ------------------------------------------------
            # Do not retry permanent/quota errors.
            # ------------------------------------------------

            if is_non_retryable_error(exc):

                print()
                print(
                    "[Gemini] Error appears to be "
                    "non-retryable."
                )

                print(
                    "[Gemini] Stopping retries."
                )

                break

            # ------------------------------------------------
            # Retry temporary failures.
            # ------------------------------------------------

            if attempt < max_retries:

                wait_time = 2 ** attempt

                print(
                    f"[Gemini] Retrying in "
                    f"{wait_time}s..."
                )

                time.sleep(wait_time)

    raise RuntimeError(
        "Gemini generation failed: "
        f"{last_error}"
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

    transcript_file = Path(
        transcript_path
    )

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

    word_count = len(
        transcript.split()
    )

    print(
        f"Transcript length: "
        f"{character_count:,} characters"
    )

    print(
        f"Transcript words:  "
        f"{word_count:,}"
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

    issues = validate_content(
        data
    )

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

    substack_path = save_substack_post(
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
        f"Substack:   {substack_path}"
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
        f"Substack chars: "
        f"{len(data.substack_content):,}"
    )

    return data


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate X, Reddit and Substack content "
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