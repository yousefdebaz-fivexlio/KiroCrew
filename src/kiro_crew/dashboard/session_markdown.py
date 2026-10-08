"""Render an export bundle as Markdown — the human-readable export format.

``GET /api/chat/slots/{slot}/export?format=md`` answers
``<title-slug>-<stamp>.kcsession.md`` instead of the gzipped JSON bundle.

**Why a second format rather than a second endpoint.** The JSON bundle is a
*transfer* document: ``session_transfer`` validates it on the way back in, so
every byte in it is there for the importer. A person reading a past conversation
needs the opposite document — one a text editor opens, a forge renders, and a
reader skims. Those are two renderings of the same assembled bundle, not two
sources of truth, so this module is a pure renderer over the bundle
``session_export`` already builds: the slot lookup, the app-scope checks, the
restricted-session refusal, the transcript locking and the egress redaction all
stay in one place and are shared.

**It installs nowhere, so it carries no Layer B.** Markdown is a one-way door:
nothing reads it back, and ``session_export`` passes ``include_layer_b=False``
for this format unconditionally — not as an operator choice. The model's context
window is byte-exact and unredacted (``session_transfer._read_layer_b``), and a
document whose whole purpose is being pasted into a review, a ticket or a chat is
the last place it may appear.

**Streamed, never assembled.** :func:`write_markdown_file` writes a message at a
time into a temp file under the export staging directory, the same way
``session_transfer.write_bundle_file`` does, because a long session's rendered
text is far too much to hold resident. :func:`iter_session_markdown` is the pure
generator behind it, which is what the tests drive.

**Message content is written verbatim.** Transcript content is already Markdown —
that is what the dashboard renders — so escaping it would destroy the fenced code
blocks, tables and lists this format exists to preserve.

Structural risks follow from that, and :func:`_unterminated_blocks` resolves them
at the message boundary. A turn ending inside an unterminated code fence swallows
every later turn into one code block; a turn ending inside an unterminated HTML
comment, or inside any other HTML block that outlives a blank line (``<pre``,
``<script``, ``<style``, ``<textarea``, a ``<?`` processing instruction, a ``<!``
declaration, or ``<![CDATA[``), hides every later turn from the renderer
entirely. Both are *silent* read-time loss — the text is in the file and no reader
sees it — which is why they are closed rather than escaped.

**The guard can cause the very loss it prevents, so it runs in both directions.**
A manufactured closer is itself content, and a bare ````` ``` ````` line is a valid
*opener*: closing a construct that was never open swallows the rest of the
document exactly as leaving a real one open does. So detection is as load-bearing
as closing, and several constructs are tracked only so that nothing is appended
for them — a fence inside a list item is already closed by the ``---`` separator's
dedent, a fence-looking line inside an HTML block is raw text that opened nothing,
and an HTML block that ends at a blank line is closed before the next turn begins.

**What verbatim content can still do, and why that is accepted.** A message body
is written at the same heading depth as :func:`_message_heading`, so content
holding a line like ``## Assistant — <ts>`` renders as a turn this session never
produced. Closing that hole means either escaping content — which destroys the
fenced blocks and tables this format exists for — or wrapping each body in a
container, which no Markdown construct provides without the same escaping. The
JSON bundle stays the fidelity document; this one is a reading copy, and a reader
who needs provenance for a specific line has the bundle. The two risks above are
different in kind: they make the file *unreadable* rather than *misleading*, and
they cost nothing to close.
"""

from __future__ import annotations

import html
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator

from kiro_crew.dashboard.session_transfer import _egress_tmp_dir, _rm_import_temps
from kiro_crew.messaging.split import fence_closes, fence_opening
from kiro_crew.widget_parse import mask_inline_code

#: Extension of an exported Markdown session. Carries the same ``.kcsession``
#: middle as the JSON export so the two sort together and a file that outlived
#: its tab is still identifiable by name alone, with the final ``.md`` telling an
#: editor, a forge and an operator how to open it.
MARKDOWN_FILE_SUFFIX = ".kcsession.md"

#: Heading shown for each role. Only ``user`` and ``assistant`` reach a bundle
#: (``session_transfer._VISIBLE_ROLES``); an unrecognised role is titled rather
#: than dropped, because losing a turn from a transcript is worse than printing a
#: role name this module did not expect.
_ROLE_LABELS = {"user": "User", "assistant": "Assistant"}

#: Provenance rows, in display order: the bundle/``source`` key and its label. A
#: field the source gateway could not read is absent from ``source`` rather than
#: empty (``session_transfer.build_source_record``), so an absent row means "not
#: known" and is skipped rather than printed blank.
_SOURCE_ROWS = (
    ("exported_at", "Exported"),
    ("model", "Model"),
    ("reasoning_effort", "Reasoning effort"),
    ("approval_policy", "Approval policy"),
    ("workspace", "Workspace"),
    ("project", "Project"),
    ("producer", "Produced by"),
)

#: What ``approval_policy`` reads as in the table. The empty string is a VALUE
#: there and not an absence — it is the interactive policy, the same spelling the
#: session object uses — so it is rendered rather than skipped, and named in
#: words because "" would read as a gateway that had nothing to report.
_INTERACTIVE_POLICY = "interactive"


def _single_line(value: str) -> str:
    """*value* with every run of whitespace collapsed to one space.

    An ATX heading is one line by definition, so a title carrying a newline does
    not render as a two-line heading — it renders as a heading plus whatever
    followed, as its own block. A title of ``Notes`` followed by a line that opens
    an HTML comment therefore hides the entire transcript beneath it, and the
    title is the one part of this document a user types freely: the rename
    endpoint trims and truncates but keeps newlines, so this is where that has to
    be handled.

    Collapsed rather than escaped, because a heading is a display line and a title
    that wanted to be two lines has nothing to express here. The provenance table
    escapes its cells instead (:func:`_escape_cell`): there the delimiter is
    structural and the text around it is data worth keeping intact.
    """
    return " ".join(value.split())


def _escape_cell(value: str) -> str:
    """Make *value* safe inside a Markdown table cell.

    A table cell is the one place in this document where content cannot be
    verbatim: an unescaped ``|`` ends the cell and shifts every later column, and
    a newline ends the row. Only provenance strings pass through here — a
    workspace name, a model id — never message content, which is written as
    ordinary block text where neither character means anything.
    """
    markdown_safe = (
        value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    )
    return html.escape(markdown_safe, quote=False)


#: An unordered marker (``-``, ``*``, ``+``) or ordered marker (``1.``, ``2)``, …)
#: followed by the required space or tab, or by the end of the line for an empty
#: item (CommonMark §5.2), used only to find where a list item's content begins
#: in :func:`_list_item_content_column` — never to validate list structure in
#: general.
_LIST_MARKER_RE = re.compile(r"(?P<marker>[-*+]|[0-9]{1,9}[.)])(?=[ \t]|$)")


def _expand_marker_tabs(line: str) -> str:
    """*line* with tabs in the whitespace after a list marker expanded to spaces.

    CommonMark §2.2 measures that padding in columns with tab stops of 4, so
    ``2.\\titem`` puts its content at column 4, not at the character after the
    tab. Expanding only that prefix lets the space-counting column logic below
    stay as it is; the content after the padding is left exactly as written.
    """
    stripped = line.lstrip(" ")
    match = _LIST_MARKER_RE.match(stripped)
    if match is None:
        return line
    end = len(line) - len(stripped) + match.end()
    rest = line[end:]
    gap = len(rest) - len(rest.lstrip(" \t"))
    if "\t" not in rest[:gap]:
        return line
    return line[: end + gap].expandtabs(4) + rest[gap:]


def _list_item_content_column(line: str, *, at_paragraph_start: bool = True) -> int | None:
    """The column a list item's content starts at, or ``None`` if *line* starts none.

    That column is the marker line's own leading spaces plus the marker and its
    required space (``"- "``, ``"* "``, ``"1. "``, …). Everything indented to it
    belongs to the item; a later non-blank line indented LESS ends the item
    (CommonMark §5.2).

    This is tracked because a fenced code block inside a list item is not a
    top-level fence, and neither is its closer — both are indented to this
    column. The scan's job is only to recognise that it is inside an item, so a
    fence there is attributed to the item and never reaches the top-level
    tracking in either direction: not as an opener, and not as a stray closer.

    The column is NOT the item's fence; a fence may appear on the marker line
    (``- ```python``) or on any later line of the item (``- Example:`` then a
    blank line then an indented fence), and the two have to behave the same. An
    earlier revision of this module keyed on the marker line alone and so missed
    the second shape, which is the more common one in a truncated reply.

    The column is where the item's content actually starts, which is the marker
    plus however much padding follows it (CommonMark §5.2) — not the marker plus
    exactly one space. ``-   Example`` puts the content at column 4, and a
    one-space-only reading would misplace a real top-level fence written at that
    column as inside the item, withholding its closer. Five or more spaces of
    padding collapse to one (the rest becomes the item's own indented content),
    which an earlier revision also did not account for. Tabs in that padding must
    already be expanded (:func:`_expand_marker_tabs`).

    A fence inside an item never draws a manufactured closer: the ``---``
    separator that follows every message is dedented past this column, which ends
    the item and closes the fence with it, so a closer here would open a *new*
    top-level fence instead. See :func:`_unterminated_blocks`.
    """
    stripped = line.lstrip(" ")
    match = _LIST_MARKER_RE.match(stripped)
    if match is None:
        return None
    marker_end = len(line) - len(stripped) + match.end()
    rest = line[marker_end:]
    padding = len(rest) - len(rest.lstrip(" "))  # includes the required space
    # Five or more spaces start indented code, and an item that starts blank puts
    # its content one column past the marker (CommonMark §5.2).
    if padding >= 5 or not rest.strip():
        padding = 1
    column = marker_end + padding
    if not at_paragraph_start:
        marker = match.group("marker")
        # CommonMark §5.2: a list may interrupt a paragraph only when its first
        # item is non-empty and, for an ordered list, starts at 1. Without this,
        # ``text\n2. item`` invents a container and hides a later top-level fence.
        if not line[column:].strip() or (marker[0] in "0123456789" and int(marker[:-1]) != 1):
            return None
    return column


def _opens_html_block_comment(line: str) -> bool:
    """Whether *line* begins a raw HTML block with ``<!--`` (CommonMark §4.6, 2).

    The condition is the line *beginning* with ``<!--`` after at most three
    spaces, which is what makes the whole line raw HTML. A ``<!--`` appearing
    mid-line is inline content instead, and an unterminated one there is not
    recognised as a comment at all — it is escaped to ``&lt;!--`` and renders as
    the literal text the author typed, so it hides nothing and needs no closer.
    """
    stripped = line.lstrip(" ")
    return len(line) - len(stripped) <= 3 and stripped.startswith("<!--")


def _next_html_comment_start(line: str, cursor: int) -> int:
    """Find ``<!--`` outside an HTML tag's quoted attributes."""
    in_tag = False
    quote = ""
    i = cursor
    while i < len(line):
        char = line[i]
        if quote:
            if char == quote:
                quote = ""
        elif in_tag:
            if char in "\"'":
                quote = char
            elif char == ">":
                in_tag = False
        elif line.startswith("<!--", i):
            return i
        elif char == "<" and (
            line[i + 1 : i + 2].isalpha()
            or (line[i + 1 : i + 2] == "/" and line[i + 2 : i + 3].isalpha())
            or (line[i + 1 : i + 2] in "!?" and line[i + 2 : i + 3].isalpha())
        ):
            in_tag = True
        i += 1
    return -1


def _mask_html_comments(line: str, inside: bool) -> tuple[str, bool]:
    """Mask HTML comments in *line* and return the resulting open state.

    Walked pair by pair rather than tested with ``"-->" in line``, because one
    line can both close a comment and open another: ``<!-- a --> <!-- b`` ends its
    HTML block at the ``-->`` yet still emits an unclosed ``<!--``. Masking keeps
    raw-block terminators inside a comment from closing their enclosing block.
    New openers are ignored inside quoted tag attributes, where they are data.
    """
    masked = list(line)
    cursor = 0
    if inside:
        end = line.find("-->")
        if end < 0:
            return " " * len(line), True
        cursor = end + 3
        masked[:cursor] = " " * cursor

    while True:
        start = _next_html_comment_start(line, cursor)
        if start < 0:
            return "".join(masked), False
        end = line.find("-->", start + 4)
        if end < 0:
            masked[start:] = " " * (len(line) - start)
            return "".join(masked), True
        cursor = end + 3
        masked[start:cursor] = " " * (cursor - start)


#: The four bare CommonMark §4.6 condition-1 tags whose HTML blocks hold
#: content opaquely across blank lines. Each ends at its matching ``</tag>``.
#: This is the MARKDOWN set: it decides which lines open a raw HTML block
#: (:func:`_opens_raw_html_block`), and nothing else.
_RAW_TEXT_TAGS = ("pre", "script", "style", "textarea")

#: The RENDERED set: elements an HTML parser reads as raw text or RCDATA until
#: their own close tag, whatever Markdown container emitted them. It is wider
#: than the CommonMark set above because the two answer different questions —
#: ``<title>`` and ``<iframe>`` open only a blank-terminated Markdown block
#: (condition 6), yet a browser still swallows every later heading as element
#: text until ``</title>`` or ``</iframe>``, so an unterminated one needs a
#: closer at the message boundary just as ``<script>`` does. ``pre`` is kept
#: for the nesting it allows (see :func:`_track_raw_text_elements`).
_RENDERED_RAW_TEXT_TAGS = (
    "pre",
    "script",
    "style",
    "textarea",
    "title",
    "iframe",
    "xmp",
    "noembed",
    "noframes",
)
#: Of those, the ones whose content an HTML parser does not parse at all, so a
#: tracked opener inside them is text and only their own close tag counts.
_OPAQUE_RENDERED_TAGS = frozenset(_RENDERED_RAW_TEXT_TAGS) - {"pre"}
#: The tag name must be followed by whitespace, ``/``, ``>`` or the END OF THE
#: LINE: an HTML tag may break its attributes across lines, so ``<script`` alone
#: on a line is as real an opener as ``<script>`` — the ``>`` arrives on a later
#: line. ``end`` is empty exactly when that happens, and the caller carries the
#: still-open tag into the next line (:func:`_track_raw_text_elements`).
_RAW_TEXT_ELEMENT_RE = re.compile(
    r"<(?P<close>/)?(?P<tag>"
    + "|".join(_RENDERED_RAW_TEXT_TAGS)
    + r")(?=[ \t/>]|$)[^>]*(?P<end>>)?",
    re.IGNORECASE,
)


def _track_raw_text_elements(
    line: str,
    open_tags: list[str],
    *,
    in_comment: bool = False,
    markdown: bool = True,
    in_tag: bool = False,
) -> bool:
    """Update rendered raw-text *open_tags* after *line*; return the open-tag state.

    Unlike a CommonMark HTML block, one of these elements may start inline or
    behind quote/list markers and survives the Markdown container that emitted
    it. ``script``, ``style``, ``textarea``, ``title``, ``iframe`` and the other
    :data:`_OPAQUE_RENDERED_TAGS` are opaque until their own close; ``pre``
    still contains parsed HTML and can therefore nest another tracked element.

    *markdown* says whether CommonMark interprets *line* at all. In ordinary
    paragraph text balanced inline code is masked, because its apparent tags
    are escaped and reach the reader as text. Inside an HTML block — a line that
    opens one, or any later line of a raw or blank-terminated block — backticks
    mean nothing and the tags between them are emitted verbatim, so the caller
    passes ``markdown=False`` and the line is scanned as written: masking there
    would hide a ``<script>`` the renderer really opens, and withhold the only
    closer it needed.

    *in_tag* is whether the previous line ended inside a tracked tag whose
    ``>`` had not yet arrived — ``<script`` on one line and ``type="…">`` on the
    next is one tag to an HTML parser, and the element it opens is as real as
    one written on a single line. The return value is that state after *line*,
    so the caller threads it through; until the ``>`` is found the text is the
    tag's own attributes and opens nothing else.
    """
    scanned = mask_inline_code(line) if markdown else line
    masked, _ = _mask_html_comments(scanned, in_comment)
    cursor = 0
    if in_tag:
        end = masked.find(">")
        if end < 0:
            return True
        cursor = end + 1
    while cursor < len(masked):
        if open_tags and open_tags[-1] in _OPAQUE_RENDERED_TAGS:
            tag = open_tags[-1]
            close = re.search(rf"</{re.escape(tag)}[ \t]*>", masked[cursor:], re.I)
            if close is None:
                return False
            cursor += close.end()
            open_tags.pop()
            continue
        match = _RAW_TEXT_ELEMENT_RE.search(masked, cursor)
        if match is None:
            return False
        cursor = match.end()
        tag = match.group("tag").lower()
        if match.group("close"):
            if tag in open_tags:
                del open_tags[len(open_tags) - 1 - open_tags[::-1].index(tag)]
        else:
            open_tags.append(tag)
        if not match.group("end"):
            return True
    return False


#: Conditions 6 and 7 instead end at a BLANK LINE, so an unterminated one is
#: already closed by the blank line before the next message's separator and needs
#: no terminator. They are still tracked, because their content is raw HTML and a
#: fence line inside one opens nothing. Condition 6's tag set, verbatim.
_BLANK_TERMINATED_TAGS = (
    "address",
    "article",
    "aside",
    "base",
    "basefont",
    "blockquote",
    "body",
    "caption",
    "center",
    "col",
    "colgroup",
    "dd",
    "details",
    "dialog",
    "dir",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "frame",
    "frameset",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "hr",
    "html",
    "iframe",
    "legend",
    "li",
    "link",
    "main",
    "menu",
    "menuitem",
    "nav",
    "noframes",
    "ol",
    "optgroup",
    "option",
    "p",
    "param",
    "search",
    "section",
    "summary",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "title",
    "tr",
    "track",
    "ul",
)

#: Condition 7: a line holding nothing but one complete open or close tag.
#: Quoted attribute values may contain ``>``; it ends a tag only outside quotes.
_LONE_TAG_RE = re.compile(
    r"""^(?:
        <[A-Za-z][A-Za-z0-9-]*
        (?:[ \t]+[A-Za-z_:][A-Za-z0-9_.:-]*
            (?:[ \t]*=[ \t]*(?:[^ \t\n\"'=<>`]+|'[^']*'|\"[^\"]*\"))?
        )*[ \t]*/?>
        |</[A-Za-z][A-Za-z0-9-]*[ \t]*>
    )[ \t]*$""",
    re.VERBOSE,
)


def _opens_raw_html_block(line: str) -> tuple[str, str] | None:
    """The raw-text HTML block *line* opens, as ``(kind, terminator)``, or ``None``.

    These are the CommonMark §4.6 conditions whose block runs past blank lines, so
    one left open at the message boundary hides every turn after it and the
    terminator has to be supplied:

    * condition 1 — ``<pre``, ``<script``, ``<style``, ``<textarea`` -> ``</tag>``
    * condition 3 — a processing instruction, ``<?`` -> ``?>``
    * condition 4 — a declaration, ``<!`` then a letter (``<!DOCTYPE``) -> ``>``
    * condition 5 — ``<![CDATA[`` -> ``]]>``

    Condition 2 (``<!--``) is the same shape and is tracked separately because a
    comment can open and close more than once on one line
    (:func:`_mask_html_comments`).

    The opener test for condition 1 requires whitespace, ``>``, or end of line
    after the tag name, which is what keeps ``<prefix>`` from reading as ``<pre``.
    Conditions 3-5 are prefix matches, and CDATA is tested before the bare
    declaration because ``<![CDATA[`` also starts with ``<!``.

    Content inside any of these is raw text, so a ````` ``` ````` line there opens
    no fence. Tracking them is therefore load-bearing in both directions: it
    supplies a missing terminator, and it stops the fence scan manufacturing a
    closer for a fence that was never open — which would itself open one.
    """
    stripped = line.lstrip(" ")
    if len(line) - len(stripped) > 3:
        return None
    lowered = stripped.lower()
    for tag in _RAW_TEXT_TAGS:
        if lowered.startswith(f"<{tag}"):
            rest = lowered[len(tag) + 1 :]
            if rest == "" or rest[0] in " \t>":
                return tag, f"</{tag}>"
    if stripped.startswith("<![CDATA["):
        return "cdata", "]]>"
    if stripped.startswith("<?"):
        return "pi", "?>"
    if stripped.startswith("<!") and stripped[2:3].isalpha():
        return "declaration", ">"
    return None


def _opens_blank_terminated_html_block(line: str, *, at_paragraph_start: bool) -> bool:
    """Whether *line* opens an HTML block that ends at the next BLANK line.

    CommonMark §4.6 conditions 6 (a known block tag) and 7 (a line holding one
    complete tag and nothing else). No terminator is ever needed for these — the
    blank line before the next message's separator already closes them — but their
    content is raw HTML, so a fence line inside one opens nothing and must not
    draw a closer. ``<div>`` then a ````` ``` ````` line is the ordinary case.

    Condition 7 additionally may not interrupt a paragraph (unlike condition 6,
    which can): *at_paragraph_start* is false while the previous line was
    non-blank text, and a lone tag there is ordinary paragraph content rather than
    a block opener — ``text`` then ``<span>`` is one paragraph, not a block, and
    treating it as one withholds a closer for a fence that follows and silently
    hides it. Erring the other way — treating a genuine condition-7 opener as
    paragraph text — only ever withholds a closer, and a withheld closer cannot
    corrupt a document the way an invented one can, which is why only this
    direction is checked.
    """
    stripped = line.lstrip(" ")
    if len(line) - len(stripped) > 3:
        return False
    lowered = stripped.lower()
    for tag in _BLANK_TERMINATED_TAGS:
        for prefix in (f"<{tag}", f"</{tag}"):
            if lowered.startswith(prefix):
                rest = lowered[len(prefix) :]
                if rest == "" or rest[0] in " \t>" or rest.startswith("/>"):
                    return True
    return at_paragraph_start and bool(_LONE_TAG_RE.match(stripped))


#: Lines that END an open paragraph without opening anything else this module
#: tracks: an ATX heading, a block quote marker, or a thematic break (CommonMark
#: §4.1-4.3). Recognised ONLY so a dedented one reads as a real list exit rather
#: than a lazy continuation — none of them can hide a later turn, so nothing else
#: here needs to know they exist.
#:
#: A setext underline is deliberately NOT here. ``===`` can only underline a
#: paragraph in its OWN container, so one dedented out of a list item is not a
#: heading at all — it is ordinary text, and therefore a lazy continuation.
#: ``---`` ends the paragraph as a *thematic break* instead, which is why the
#: back-referenced branch below requires three of the character: ``--`` is as
#: lazy as ``===``.
_PARAGRAPH_END_RE = re.compile(
    r"""^[ ]{0,3}(?:
        \#{1,6}(?:[ \t]|$)                   # ATX heading
        |>                                    # block quote
        |([-*_])(?:[ \t]*\1){2,}[ \t]*$       # thematic break, spaced forms too
    )""",
    re.VERBOSE,
)
#: A setext underline in its OWN container (CommonMark §4.3): ``=`` for an h1,
#: ``-`` for an h2. Both end the paragraph above them. A hyphen run of three or
#: more is also a thematic break, but after paragraph text the setext reading
#: wins — and either way the paragraph is over, which is all that is asked here.
#: Only consulted while a paragraph is open: at a block start ``--`` is text
#: and ``-`` is an empty list item.
_SETEXT_UNDERLINE_RE = re.compile(r"^[ ]{0,3}(?:=+|-+)[ \t]*$")


def _interrupts_a_paragraph(line: str) -> bool:
    """Whether *line* starts a block, so it cannot LAZILY continue a paragraph.

    CommonMark §4.8: a paragraph continuation line may be indented less than its
    container requires — a *lazy* continuation — but only while it stays ordinary
    text. A line that starts a block ends the paragraph instead, and with it the
    list item that paragraph was in.

    Asked only of a dedented line inside an open list item
    (:func:`_unterminated_blocks`), where the answer decides whether that item is
    still open. Both directions matter there: a block start read as lazy text
    keeps a dead item open and withholds the closer a genuinely top-level fence
    needs, while lazy text read as a block start ends the item early and makes the
    item's own fence look top-level — drawing a column-0 closer that opens code
    across every later turn.

    An interrupting LIST MARKER is deliberately absent: the caller has already
    matched one before reaching here. ``at_paragraph_start=False`` is what
    excludes §4.6 condition 7, the one HTML block kind that cannot interrupt a
    paragraph; an indented code block is excluded by the same CommonMark rule, so
    four spaces after paragraph text stays continuation rather than a new block.
    """
    return (
        fence_opening(line) is not None
        or _opens_html_block_comment(line)
        or _opens_raw_html_block(line) is not None
        or _opens_blank_terminated_html_block(line, at_paragraph_start=False)
        or _PARAGRAPH_END_RE.match(line) is not None
    )


def _blockquote_prefix(line: str) -> tuple[str, str]:
    """Return a leading blockquote prefix and its contained Markdown.

    A closer appended outside the quote is ordinary text, so an unterminated
    comment that began in a quote needs its quote markers copied onto the closer.
    """
    cursor = 0
    while True:
        remaining = line[cursor:]
        spaces = len(remaining) - len(remaining.lstrip(" "))
        if spaces > 3 or remaining[spaces : spaces + 1] != ">":
            break
        cursor += spaces + 1
        if line[cursor : cursor + 1] in (" ", "\t"):
            cursor += 1
    return line[:cursor], line[cursor:]


def _container_steps(line: str) -> list[tuple[str, int]]:
    """The containers *line* opens, outermost first, as ``(kind, column)`` steps.

    ``("quote", 0)`` is a block quote marker; ``("list", column)`` is a list item
    whose content starts *column* characters into whatever the enclosing steps
    left. This is :func:`_container_prefix`'s walk with its shape kept instead
    of its text, so a later line can be asked whether it is still INSIDE those
    containers (:func:`_continues_containers`) — which the prefix text alone
    cannot answer, because a list item is continued by indentation and not by
    repeating its marker.
    """
    steps: list[tuple[str, int]] = []
    remaining = line
    while remaining:
        quote, quoted = _blockquote_prefix(remaining)
        if quote:
            steps.append(("quote", 0))
            remaining = quoted
            continue
        expanded = _expand_marker_tabs(remaining)
        column = _list_item_content_column(expanded)
        if column is None:
            break
        steps.append(("list", column))
        remaining = expanded[column:]
    return steps


def _continues_containers(line: str, steps: list[tuple[str, int]]) -> bool:
    """Whether *line* is still inside every container in *steps*.

    Asked of the lines after a comment that opened inside a block quote. An HTML
    block cannot be lazily continued (CommonMark §4.6 and §5.1), so the quote —
    and the comment's hold on the Markdown — ends at the first line that does
    not carry its marker: a quote step needs its ``>`` again, and a list step
    needs the line indented to its content column, or blank, which a list item
    survives. Anything less ends the container, and with it the HTML block.
    """
    remaining = line
    for kind, column in steps:
        if kind == "quote":
            quote, remaining = _blockquote_prefix(remaining)
            if not quote:
                return False
            continue
        if not remaining.strip():
            continue
        if len(remaining) - len(remaining.lstrip(" ")) < column:
            return False
        remaining = remaining[column:]
    return True


def _container_prefix(line: str) -> tuple[str, str]:
    """Return every leading quote/list marker and the contained Markdown."""
    prefix = ""
    remaining = line
    while remaining:
        quote, quoted = _blockquote_prefix(remaining)
        if quote:
            prefix += quote
            remaining = quoted
            continue
        expanded = _expand_marker_tabs(remaining)
        column = _list_item_content_column(expanded)
        if column is None:
            break
        prefix += expanded[:column]
        remaining = expanded[column:]
    return prefix, remaining


def _unterminated_blocks(content: str) -> str:
    """The text needed to close whatever *content* leaves open, or ``""``.

    Message content is written verbatim, so a turn that ends inside a code fence
    would make every following role heading render as code, and a turn that ends
    inside an HTML comment or a raw HTML block would hide them from the renderer
    altogether. Either way the whole rest of the conversation disappears at read
    time while staying present in the file, which is the failure mode worth paying
    a scan for. Returning the closing text lets the caller terminate the block at
    the message boundary, which costs a balanced message nothing.

    A manufactured closer is itself content, so emitting one that was not needed
    is not a harmless no-op: a bare ````` ``` ````` is a *valid opener*, so closing a
    fence that was never open swallows the rest of the document — the very loss
    being prevented. Every state below is therefore tracked to decide BOTH things,
    and some are tracked only to stay silent.

    Most states are mutually exclusive by construction: inside a fence,
    ``<!--`` is literal text that CommonMark escapes, and inside a standalone
    comment a ````` ``` ````` line is raw HTML rather than a fence. A comment
    emitted inside raw or blank-terminated HTML is the exception: the Markdown
    container can end while its HTML comment remains open, so the scanner keeps
    that comment only long enough to append an effective closer after any later
    Markdown fence.

    **Which HTML blocks need a terminator is decided by whether they survive a
    blank line, not by how common they are.** CommonMark §4.6 conditions 1-5 —
    ``<pre``/``<script``/``<style``/``<textarea``, ``<!--``, ``<?``, a ``<!``
    declaration, and ``<![CDATA[`` — all run until their own terminator, past any
    number of blank lines, so one left open hides every following turn and the
    terminator is supplied (:func:`_opens_raw_html_block`). Conditions 6 and 7 end
    at the next blank line, so the blank line before the next message's separator
    has already closed them and nothing is appended — but they are still tracked,
    because their content is raw HTML and a fence line inside one opens nothing
    (:func:`_opens_blank_terminated_html_block`).

    **A list item is tracked as a container, not by its marker line.** A fence may
    open on the marker line (``- ```python``) or on any later line of the item, and
    both are the item's fence rather than a top-level one. So the scan follows the
    item's content column and attributes everything indented to it to the item
    (:func:`_list_item_content_column`). A fence there never draws a closer: the
    ``---`` separator that follows every message is dedented past that column,
    which ends the item and closes the fence with it, so a closer here would open a
    *new* top-level fence and cause exactly the loss this function prevents. A
    non-blank line dedented past the column ends the item and is then read as
    ordinary top-level content, so a fence opened after a list is still seen —
    unless it is a LAZY CONTINUATION of a paragraph still open inside the item,
    which CommonMark §4.8 allows to be dedented and which therefore leaves the
    item open (:func:`_interrupts_a_paragraph`).

    An info string disqualifies a line from CLOSING a fence (CommonMark forbids
    one there), which is why the open fence's own trailing text is ignored but a
    candidate closer's is not.
    """
    open_char = ""
    open_len = 0
    in_comment = False
    raw_terminator = ""
    raw_indent = 0
    raw_blocks_markdown = False
    output_raw_tags: list[str] = []
    in_blank_terminated = False
    blank_indent = 0
    list_columns: list[int] = []
    empty_item_depth = 0
    list_fence_char = ""
    list_fence_len = 0
    comment_indent = 0
    comment_prefix = ""
    comment_list_depth = 0
    comment_blocks_markdown = False
    # The quote/list containers a QUOTED comment opened inside, so the lines
    # after it can be asked whether that quote is still open. Empty for a
    # comment that began anywhere else.
    comment_steps: list[tuple[str, int]] = []
    # Whether the previous line ended inside a tracked HTML tag whose ">" has
    # not arrived yet (:func:`_track_raw_text_elements`).
    raw_in_tag = False
    # Block quotes are not tracked as containers — a fence inside one is closed
    # by the quote ending, so nothing in a quote draws a closer. What IS tracked
    # is the one way a quote reaches past its own lines: a paragraph open inside
    # it may be LAZILY continued by an unquoted line (CommonMark §5.1). That
    # line belongs to the quote, not to the top level: it opens no paragraph
    # there, and a list or HTML block on the line after it starts freely — the
    # quote did not match that line, so there is no paragraph to interrupt.
    # ``quote_fence`` keeps a fence inside the quote from reading as a paragraph.
    paragraph_in_quote = False
    quote_fence: tuple[str, int] | None = None
    quote_fence_depth = 0
    at_paragraph_start = True
    for raw_line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        # CommonMark measures leading indentation in columns with tab stops of 4;
        # the column logic below counts spaces, so spell a leading tab as spaces.
        body_start = len(raw_line) - len(raw_line.lstrip(" \t"))
        line = raw_line[:body_start].expandtabs(4) + raw_line[body_start:]
        if open_char:
            # Inside a fenced code block nothing else applies: CommonMark escapes
            # the content, so a comment opener there is literal text, and only a
            # long-enough run of the same character ends the block.
            fence = fence_opening(line)
            if fence is not None and fence_closes(line, open_char, open_len):
                open_char, open_len = "", 0
            at_paragraph_start = True
            continue

        if list_fence_char:
            # An active list-item fence owns every sufficiently indented line.
            list_column = list_columns[-1] if list_columns else 0
            indent = len(line) - len(line.lstrip(" "))
            if list_column and (not line.strip() or indent >= list_column):
                if fence_closes(line[list_column:], list_fence_char, list_fence_len):
                    list_fence_char, list_fence_len = "", 0
                at_paragraph_start = True
                continue
            # Dedenting out of an unterminated list fence closes it with the
            # message separator, so resume normal container detection here.
            list_fence_char, list_fence_len = "", 0

        indent = len(line) - len(line.lstrip(" "))
        if empty_item_depth:
            # An empty item holds no paragraph to continue lazily, so it ends at
            # a blank line or at any line not indented to its content column.
            if not line.strip() or (list_columns and indent < list_columns[empty_item_depth - 1]):
                list_columns = list_columns[: empty_item_depth - 1]
                at_paragraph_start = True
            empty_item_depth = 0
        if list_columns and line.strip() and indent < list_columns[-1]:
            if raw_blocks_markdown and raw_indent >= list_columns[-1]:
                raw_blocks_markdown = False
            if in_blank_terminated and blank_indent >= list_columns[-1]:
                in_blank_terminated, blank_indent = False, 0
            if in_comment and comment_blocks_markdown:
                comment_blocks_markdown = False

        # A comment that began inside a BLOCK QUOTE owns the Markdown only while
        # the quote lasts: an HTML block cannot be lazily continued, so the first
        # line without the quote's marker ends the quote and the HTML block with
        # it, and that line is ordinary top-level Markdown again. The comment is
        # still open in the rendered output — the closer is still owed — but a
        # fence opened on this line or later is a real fence, and the closer has
        # to land after it rather than inside it, where CommonMark would escape
        # its "-->" and leave every later turn hidden.
        if (
            in_comment
            and comment_blocks_markdown
            and comment_steps
            and not _continues_containers(line, comment_steps)
        ):
            comment_blocks_markdown = False
            comment_steps = []
            comment_prefix = ""
            comment_indent = 0
            comment_list_depth = 0

        # A comment that began inside raw or blank-terminated HTML outlives that
        # Markdown container in rendered HTML, but does not keep consuming
        # Markdown after the container closes. Standalone comments do consume it.
        if (
            (raw_terminator and raw_blocks_markdown)
            or in_blank_terminated
            or (in_comment and comment_blocks_markdown)
        ):
            was_in_comment = in_comment
            active_indent = raw_indent if raw_terminator else blank_indent
            # Every line here is raw HTML, so backticks are text and the tags
            # between them are emitted as written.
            raw_in_tag = _track_raw_text_elements(
                line, output_raw_tags, in_comment=in_comment, markdown=False, in_tag=raw_in_tag
            )
            _, in_comment = _mask_html_comments(line, in_comment)
            if in_comment and not was_in_comment:
                comment_indent = active_indent
                comment_prefix = ""
                comment_list_depth = len(list_columns)
                comment_blocks_markdown = False
                comment_steps = []
            # CommonMark ends raw HTML at its terminator even when an HTML
            # comment in the emitted HTML remains open.
            if raw_terminator and raw_terminator in line.lower():
                raw_terminator, raw_indent, raw_blocks_markdown = "", 0, False
            if in_blank_terminated and not line.strip():
                in_blank_terminated, blank_indent = False, 0
            at_paragraph_start = True
            continue

        comment_prefix_candidate, comment_candidate = _container_prefix(line)
        if ">" in comment_prefix_candidate and _opens_html_block_comment(comment_candidate):
            _, in_comment = _mask_html_comments(comment_candidate, False)
            comment_indent = 0
            comment_prefix = comment_prefix_candidate
            comment_list_depth = 0
            comment_blocks_markdown = True
            comment_steps = _container_steps(line)
            paragraph_in_quote = False
            at_paragraph_start = True
            continue

        quote_depth = comment_prefix_candidate.count(">")
        if quote_depth:
            # A quoted line. Its innermost content decides whether the quote
            # leaves a paragraph open for an unquoted line to continue lazily.
            if quote_fence is not None and quote_depth < quote_fence_depth:
                quote_fence = None
            if quote_fence is not None:
                if fence_closes(comment_candidate, *quote_fence):
                    quote_fence = None
                paragraph_in_quote = False
            elif (quoted_fence := fence_opening(comment_candidate)) is not None:
                quote_fence, quote_fence_depth = quoted_fence, quote_depth
                paragraph_in_quote = False
            else:
                paragraph_in_quote = bool(
                    comment_candidate.strip()
                    and not comment_candidate.startswith("    ")
                    and not _interrupts_a_paragraph(comment_candidate)
                    and not _opens_blank_terminated_html_block(
                        comment_candidate, at_paragraph_start=True
                    )
                )
        elif paragraph_in_quote or quote_fence is not None:
            # An unquoted line ends the quote — unless it is plain text, which
            # lazily continues the quote's open paragraph. A block start of ANY
            # kind ends it: the quote did not match this line, so its paragraph
            # is not the container here and cannot be "interrupted", and a list
            # marker that could not interrupt a paragraph starts a list. A
            # setext-looking line is NOT a block start: an underline cannot be
            # lazy (CommonMark §4.3, example 94), so it stays paragraph text
            # inside the quote and the quote stays open.
            quote_fence = None
            if (
                paragraph_in_quote
                and line.strip()
                and not _interrupts_a_paragraph(line)
                and not _opens_blank_terminated_html_block(line, at_paragraph_start=True)
                and _list_item_content_column(_expand_marker_tabs(line)) is None
            ):
                # Lazy continuation: the line belongs to the quote's paragraph,
                # opens nothing at the top level, and leaves every open list
                # container exactly as it was. Its inline HTML still renders.
                raw_in_tag = _track_raw_text_elements(
                    line, output_raw_tags, in_comment=in_comment, in_tag=raw_in_tag
                )
                continue
            paragraph_in_quote = False

        expanded = _expand_marker_tabs(line)
        indent = len(line) - len(line.lstrip(" "))
        # A nested item ends at its own content column; restore the enclosing
        # item instead of discarding every list container.
        while len(list_columns) > 1 and line.strip() and indent < list_columns[-1]:
            list_columns.pop()
        list_column = list_columns[-1] if list_columns else 0
        dedented_from_item = bool(list_column and line.strip() and indent < list_column)
        # A spaced thematic break is a block boundary, not a bullet marker.
        thematic_break = _PARAGRAPH_END_RE.match(line) is not None
        marker_column = (
            None
            if thematic_break
            else _list_item_content_column(
                expanded, at_paragraph_start=at_paragraph_start or dedented_from_item
            )
        )
        if marker_column is not None:
            marker_indent = len(expanded) - len(expanded.lstrip(" "))
            if not list_columns:
                list_columns = [marker_column]
            elif marker_indent < list_columns[-1]:
                list_columns = [marker_column]
            else:
                list_columns.append(marker_column)
            list_column = list_columns[-1]
            block_line = expanded[list_column:]
            block_indent = list_column
            in_list_item = True
        elif list_column and (not line.strip() or indent >= list_column):
            block_line = line[list_column:]
            block_indent = list_column
            in_list_item = True
        elif list_column and not at_paragraph_start and not _interrupts_a_paragraph(line):
            # Lazy continuation remains in the current list item, and its inline
            # HTML still reaches the rendered output.
            raw_in_tag = _track_raw_text_elements(
                line, output_raw_tags, in_comment=in_comment, in_tag=raw_in_tag
            )
            continue
        else:
            list_columns.clear()
            block_line = line
            block_indent = 0
            in_list_item = False

        if marker_column is not None and not block_line.strip():
            empty_item_depth = len(list_columns)
        indented_code = bool(
            not in_list_item
            and at_paragraph_start
            and (line.startswith("    ") or line.startswith("\t"))
        )
        opens_comment = _opens_html_block_comment(block_line)
        raw = _opens_raw_html_block(block_line)
        opens_blank_terminated = _opens_blank_terminated_html_block(
            block_line,
            at_paragraph_start=at_paragraph_start or marker_column is not None,
        )
        if not indented_code:
            # A line that opens an HTML block is raw HTML from its first
            # character, so its backticks are text rather than inline code and
            # a tag between them is really emitted; any other line is Markdown.
            raw_in_tag = _track_raw_text_elements(
                line,
                output_raw_tags,
                in_comment=in_comment,
                markdown=not (opens_comment or raw is not None or opens_blank_terminated),
                in_tag=raw_in_tag,
            )
            if raw_terminator and raw_terminator in line.lower():
                raw_terminator, raw_indent, raw_blocks_markdown = "", 0, False

        if opens_comment:
            _, in_comment = _mask_html_comments(block_line, False)
            comment_indent = block_indent
            comment_prefix = ""
            comment_list_depth = len(list_columns)
            comment_blocks_markdown = True
            comment_steps = []
            at_paragraph_start = True
            continue

        block_started = False
        fence = fence_opening(block_line)
        if fence is not None:
            block_started = True
            if in_list_item:
                list_fence_char, list_fence_len = fence
            else:
                open_char, open_len = fence
        elif raw is not None:
            block_started = True
            # The opener line itself may contain a terminator. An HTML comment in
            # that emitted raw text remains an output-level comment even though
            # CommonMark still ends the raw block at its terminator. Pass the
            # CURRENT in_comment state through: a comment already open from an
            # earlier line (e.g. a prior <div> whose own blank-terminated block
            # already ended) stays open across this new raw block, which is a
            # distinct Markdown container but not a distinct rendered comment —
            # resetting unconditionally here is what drops that still-open state.
            _kind, terminator = raw
            was_in_comment = in_comment
            _, in_comment = _mask_html_comments(block_line, in_comment)
            raw_terminator = "" if terminator in block_line.lower() else terminator
            raw_indent = block_indent
            raw_blocks_markdown = bool(raw_terminator)
            if in_comment and not was_in_comment:
                comment_indent = block_indent
                comment_prefix = ""
                comment_list_depth = len(list_columns)
                comment_blocks_markdown = False
                comment_steps = []
        elif opens_blank_terminated:
            block_started = True
            in_blank_terminated = True
            blank_indent = block_indent
            was_in_comment = in_comment
            _, in_comment = _mask_html_comments(block_line, in_comment)
            if in_comment and not was_in_comment:
                comment_indent = block_indent
                comment_prefix = ""
                comment_list_depth = len(list_columns)
                comment_blocks_markdown = False
                comment_steps = []
        # An EMPTY list item marker (``-`` alone) holds no paragraph, so the next
        # line is at a block start: a nested marker there opens a nested item
        # rather than failing to "interrupt" a paragraph that does not exist,
        # and a dedented text line is a new paragraph, not a lazy continuation.
        at_paragraph_start = bool(
            block_started
            or indented_code
            or not line.strip()
            or (marker_column is not None and not block_line.strip())
            or _PARAGRAPH_END_RE.match(block_line)
            or (not at_paragraph_start and _SETEXT_UNDERLINE_RE.match(block_line))
        )
    closers = []
    if open_char:
        closers.append(open_char * open_len)
    if in_comment:
        prefix = comment_prefix
        if not prefix and comment_list_depth <= len(list_columns):
            prefix = " " * comment_indent
        # A bare "-->" is only a real closer while it lands INSIDE the Markdown
        # construct that was open when the comment started (a list item's HTML
        # block, a blockquote's HTML block, or active raw/blank-terminated HTML).
        # Once that construct has itself ended in Markdown — the common case,
        # since the comment usually outlives it — the same text is an ordinary
        # paragraph, which CommonMark escapes to "--&gt;": the rendered HTML
        # comment stays open and swallows everything after it, including every
        # later role heading. "<!-- -->" is a complete, self-contained HTML
        # comment: wherever it lands, an HTML parser reads its own "-->" as the
        # terminator of the STILL-OPEN comment (comments do not nest), closing
        # it regardless of which Markdown construct surrounds this line.
        closers.append(prefix + "<!-- -->")
    raw_output_index = next(
        (
            index
            for index, tag in enumerate(output_raw_tags)
            if raw_terminator.lower() == f"</{tag}>"
        ),
        None,
    )
    if raw_terminator and raw_output_index is None:
        closers.append(" " * raw_indent + raw_terminator)
    for index in range(len(output_raw_tags) - 1, -1, -1):
        output_raw_terminator = f"</{output_raw_tags[index]}>"
        prefix = " " * raw_indent if index == raw_output_index else ""
        closers.append(prefix + output_raw_terminator)
    return "\n".join(closers)


def _message_heading(message: dict[str, Any]) -> str:
    """``## User`` / ``## Assistant``, with the message's timestamp when it has one.

    The timestamp is printed as the transcript recorded it — an ISO 8601 instant
    with its offset — rather than reformatted for a locale: this file is read on
    whatever machine it lands on, and an unambiguous instant survives that trip
    where a localised one does not. A row with no ``ts`` gets a bare heading
    rather than a placeholder.

    Both the label and the timestamp go through :func:`_single_line` for the reason
    the title does: a heading is one line, so a newline in either would render as a
    heading plus a separate block, and a block opening an HTML comment hides every
    turn beneath it. The gateway's own writer only ever stamps ISO instants here
    (``history.monotonic_transcript_ts``), but an IMPORTED bundle's ``ts`` is
    accepted as any string, and an unexpected role reaches ``label`` as its own
    text — so neither is this module's to trust.
    """
    role = str(message.get("role", "") or "")
    label = html.escape(
        _single_line(_ROLE_LABELS.get(role, role.title() or "Message")), quote=False
    )
    ts = html.escape(_single_line(str(message.get("ts", "") or "")), quote=False)
    return f"## {label} — {ts}\n" if ts else f"## {label}\n"


def iter_session_markdown(bundle: dict[str, Any]) -> Iterator[str]:
    """Yield the Markdown document for *bundle*, a chunk at a time. Pure.

    One chunk per heading, per content body and per separator rather than one per
    document, so :func:`write_markdown_file` never holds more than a single
    message's text. The caller joins or writes them; nothing here touches disk.

    *bundle* is the already-assembled, already-redacted export bundle — its
    ``title`` and non-user ``content`` went through ``session_transfer``'s egress
    scrubbers on the way in, and this renderer adds no new source of text.
    """
    title = html.escape(_single_line(str(bundle.get("title", "") or "")), quote=False)
    yield f"# {title}\n" if title else "# Session\n"

    source = bundle.get("source")
    rows: list[tuple[str, str]] = []
    agent = str(bundle.get("agent", "") or "")
    if agent:
        rows.append(("Agent", agent))
    if isinstance(source, dict):
        for key, label in _SOURCE_ROWS:
            if key not in source:
                continue
            value = source[key]
            if key == "approval_policy" and value == "":
                value = _INTERACTIVE_POLICY
            text = str(value)
            if text:
                rows.append((label, text))
    messages = bundle.get("messages") or []
    rows.append(("Messages", str(len(messages))))

    yield "\n| Field | Value |\n| --- | --- |\n"
    for label, value in rows:
        yield f"| {_escape_cell(label)} | {_escape_cell(value)} |\n"

    for message in messages:
        yield "\n---\n\n"
        yield _message_heading(message)
        content = str(message.get("content", "") or "")
        if not content.strip():
            continue
        yield "\n"
        yield content if content.endswith("\n") else content + "\n"
        closing = _unterminated_blocks(content)
        if closing:
            yield closing + "\n"


def write_markdown_file(bundle: dict[str, Any]) -> Path:
    """Render *bundle* to a temp Markdown file and return its path. **Blocking.**

    Staged in the export directory beside the JSON path's own temp file, so both
    formats are swept by the same orphan reclaim, and written through
    ``mkstemp`` so two concurrent exports cannot collide on a name. The caller
    owns the file and removes it with ``_rm_import_temps``; a failure mid-write
    removes it here rather than leaving a partial document behind.

    ``newline=""`` keeps the ``\\n`` line endings this renderer emits on every
    platform: Python's text mode would otherwise translate them to CRLF on
    Windows, which changes the bytes of a document whose content was captured
    with LF endings.

    ``errors="backslashreplace"`` is what keeps a transcript holding a lone
    surrogate exportable. Such a character reaches a transcript through the JSON
    wire format, which can carry an unpaired ``\\uD800`` that UTF-8 cannot encode;
    the JSON export survives it because ``ensure_ascii`` re-escapes it, so
    defaulting to ``strict`` here would make Markdown the one format that fails on
    a session the other exports. The escape is visible and reversible in the
    document, which a replacement character would not be.
    """
    fd, name = tempfile.mkstemp(dir=str(_egress_tmp_dir()), suffix=MARKDOWN_FILE_SUFFIX)
    path = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="", errors="backslashreplace") as out:
            for chunk in iter_session_markdown(bundle):
                out.write(chunk)
    except BaseException:
        _rm_import_temps(path)
        raise
    return path
