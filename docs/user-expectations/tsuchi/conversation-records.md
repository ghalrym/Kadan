# Stored conversations

Tsuchi's Conversations view manages saved records. Creating a conversation or appending a message there stores data; it does not queue an agent response or contact Hikari. Use Kadan chat to ask an agent to act or reply.

A new message defaults to the user role. The editor also permits system, assistant and tool roles, text, file attachments, stored-resource links and source links. Messages receive identifiers and timestamps when appended. The conversation must be saved before messages can be appended. Referenced files and resources must belong to the same namespace.

Appending checks the conversation's version. A conflicting edit produces an error rather than silently replacing the newer conversation. Saved messages remain ordered records; editing them in storage does not replay the corresponding tool action or inference.

Tsuchi can hold several conversation records in a namespace. Kadan currently presents one continuous chat and opens the oldest active conversation for that agent. Creating an additional conversation in Tsuchi does not add a thread selector to Kadan. Trashing or restoring storage records can therefore affect which conversation Kadan opens; preserve the active chat unless that change is intended.
