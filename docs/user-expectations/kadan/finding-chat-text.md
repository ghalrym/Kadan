# Finding chat text

Kadan's **Find in conversation** control searches the current agent's saved conversation text, including older messages behind **Show earlier messages**. Matching ignores letter case and treats punctuation literally. Leading and trailing spaces in the query are ignored. Each matching message counts once, even when it contains the text more than once.

The count shows the selected matching message. Previous and Next move through messages in conversation order and wrap at either end. Enter goes to the next match; Shift+Enter goes to the previous one. Kadan reveals older messages when needed and outlines the selected message. An empty query selects nothing, and a query with no matches says so.

Search uses saved message text, including Markdown source. It does not inspect attachment contents, resource titles, drafts, unsent submissions, retired tool activity, other conversations or previous revisions. New saved messages update the results. A selected match stays selected while it still matches, and automatic scrolling to new replies pauses while a query is active.

Escape or Close find clears the query and returns focus to Find. Closing leaves revealed messages available without jumping to the newest message. Changing agents or conversations resets search. Searching does not change the draft, unsent recovery or saved conversation.
