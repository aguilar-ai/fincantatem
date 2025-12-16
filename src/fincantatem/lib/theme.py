from typing import Final
from .ports.display import DisplaySettings

# ═══════════════════════════════════════════════════════════════════════════════
# Harry Potter-Inspired Theming for Finite Incantatem
# The counter-spell that ends enchantments - here it ends confusion around errors
# ═══════════════════════════════════════════════════════════════════════════════

# ASCII Art Headers
SPELL_HEADER_RICH: Final[
    str
] = r"""
▗▄▄▄▖    ▗▄▄▄▖▄▄▄▄  ▗▞▀▘▗▞▀▜▌▄▄▄▄     ■  ▗▞▀▜▌   ■  ▗▞▀▚▖▄▄▄▄  
▐▌         █  █   █ ▝▚▄▖▝▚▄▟▌█   █ ▗▄▟▙▄▖▝▚▄▟▌▗▄▟▙▄▖▐▛▀▀▘█ █ █ 
▐▛▀▀▘      █  █   █          █   █   ▐▌         ▐▌  ▝▚▄▄▖█   █ 
▐▌       ▗▄█▄▖                       ▐▌         ▐▌             
                                     ▐▌         ▐▌             
"""

SPELL_HEADER_PLAIN: Final[
    str
] = r"""
▗▄▄▄▖    ▗▄▄▄▖▄▄▄▄  ▗▞▀▘▗▞▀▜▌▄▄▄▄     ■  ▗▞▀▜▌   ■  ▗▞▀▚▖▄▄▄▄  
▐▌         █  █   █ ▝▚▄▖▝▚▄▟▌█   █ ▗▄▟▙▄▖▝▚▄▟▌▗▄▟▙▄▖▐▛▀▀▘█ █ █ 
▐▛▀▀▘      █  █   █          █   █   ▐▌         ▐▌  ▝▚▄▄▖█   █ 
▐▌       ▗▄█▄▖                       ▐▌         ▐▌             
                                     ▐▌         ▐▌             
"""


# Semantic Color Tokens
# These map to Rich color names for the RichTextInterface
class Colors:
    """Semantic color tokens for consistent theming."""

    SPELL: Final[str] = "bright_yellow"  # Brand accent - golden like wand sparks
    ERROR: Final[str] = "red"  # Traceback, failure states
    INSIGHT: Final[str] = "magenta"  # AI analysis, explanations
    ACTION: Final[str] = "green"  # Fixes, suggestions, commands
    NEUTRAL: Final[str] = "white"  # Borders, secondary text
    INFO: Final[str] = "cyan"  # Informational messages


# Plain text fallbacks (dependency-free markers)
class PlainMarkers:
    """Text markers for PlainTextInterface theming."""

    SPELL: Final[str] = "✧"
    ERROR: Final[str] = "!"
    INSIGHT: Final[str] = "~"
    ACTION: Final[str] = ">"
    DIVIDER: Final[str] = "-" * 60


# Chat command help text
CHAT_HELP_TEXT: Final[
    str
] = """
  ✧ Available Commands:
    /help  - Show this help message
    /save  - Save chat history as JSON
    /quit  - Exit chat (or /q)

  Or just type your follow-up question.
"""

CHAT_HELP_TEXT_PLAIN: Final[
    str
] = """
  Available Commands:
    /help  - Show this help message
    /save  - Save chat history as JSON
    /quit  - Exit chat (or /q)

  Or just type your follow-up question.
"""

# Action hint shown after analysis
ACTION_HINT_RICH: Final[str] = "  ✧ Type a follow-up question, or /help for commands"
ACTION_HINT_PLAIN: Final[str] = "  > Type a follow-up question, or /help for commands"


# Display Settings for different contexts
TRACEBACK_STYLE: DisplaySettings = {
    "box_style": "single",
    "color": "red",
    "title": "Traceback",
    "markdown_disabled": True,
    "collapsible": True,  # Show condensed traceback by default
}

ANALYSIS_STYLE: DisplaySettings = {
    "box_style": "rounded",
    "color": "magenta",
    "title": "✧ F. Incantatem",
}

CHAT_RESPONSE_STYLE: DisplaySettings = {
    "box_style": "rounded",
    "color": "magenta",
    "title": "✧ F. Incantatem",
}

CHAT_INFO_STYLE: DisplaySettings = {
    "box_style": "rounded",
    "color": "cyan",
}

CHAT_HELP_STYLE: DisplaySettings = {
    "box_style": "rounded",
    "color": "bright_yellow",
    "title": "✧ Help",
    "markdown_disabled": True,
}

USER_PROMPT_STYLE: DisplaySettings = {
    "box_style": "rounded",
    "color": "yellow",
    "title": "You",
}
