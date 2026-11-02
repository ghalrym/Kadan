# Optional progress notices

On this branch, Kadan's assistant can offer an optional routine progress notice
when a saved goal's timer wakes it. A separate decision specialist may omit a
notice it judges clearly redundant. The capable assistant continues to own the
goal, its progress, next step and waits. This proposal has unit coverage; model
quality, speed and the complete application flow remain for manual validation.

Ordinary replies, questions and goal checkpoint messages are always delivered
through their existing paths. The optional notice defaults to important and
requested by the user, so it is delivered unless the assistant explicitly marks
it routine and unsolicited. Direct user requests, tool-completion events,
failures, blocked goals, resumption and notices marked important or requested
bypass the specialist. This is a narrow optional path, not a filter on all chat.

If the specialist is unavailable, takes too long, or returns an uncertain or
invalid answer, Kadan sends the notice. It uses the existing decision capability
without automatically downloading a model or switching to another one. Its
judgments are not verified facts, and a confidence value is not a guarantee that
an omitted notice was unimportant. The assistant can also mislabel a notice;
routine classification still needs manual evaluation.

Kadan saves the notice's send-or-omit decision before acting on it. Recovery reuses
that saved outcome instead of asking the specialist again, and existing delivery
receipts prevent a delivered message from being repeated. Suppressing a notice
does not pause, cancel, complete or change the goal. Goal status remains visible
in [Goals](durable-goals.md).
