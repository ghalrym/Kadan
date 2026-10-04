# Kadan

## Overview

Kadan aims to give local agents chat, decisions, image and video generation, and
speech tools on one machine without an expensive home lab.

The server will manage model loading for each request. Its planned FreeToken-style
mixture-of-experts loading shares GPU VRAM and system RAM across models. When an
image, video, or speech request needs the GPU, Kadan can move the LLM entirely out
of VRAM and reload it afterward. Agents just call the API. Kadan handles the memory.

The goal is to fit all these tools in one box, accepting slower model switches
to keep hardware requirements down.

![Kadan request dashboard showing AI requests, latency, and GPU and system memory usage](docs/images/requests.png)

![Kadan settings showing model selections for language, images, video, and speech](docs/images/settings.png)

## Setup with Docker Compose

Install Docker with Docker Compose and run these commands from the repository
root.

1. Build and start the services:

   ```sh
   docker compose up --build -d
   ```

2. Open the app at [http://localhost:5173](http://localhost:5173).
   API docs are at [http://localhost:8000/docs](http://localhost:8000/docs).

3. Check service status or follow startup logs:

   ```sh
   docker compose ps
   docker compose logs -f
   ```

4. Stop the services when finished:

   ```sh
   docker compose down
   ```

## Publish draft pull requests as Ai Kadan

The manual **Publish draft PR as Ai Kadan** workflow opens a draft against
`master` from an existing `codex/name` branch in `ghalrym/Kadan`. Codex continues
committing and pushing through its existing connection. Only PR creation uses
the Ai Kadan installation token: this does **not** change commit authors,
committers, push identity, or authorship of existing PRs. The bootstrap PR for
this workflow is created through the existing connection too.

### One-time owner setup (after reviewing and merging this workflow)

1. Confirm Ai Kadan (App ID `5188232`, client ID `Iv23liGu3LqUPYzPqdag`,
   installation `167899795`) is installed for **Kadan only** (repository ID
   `1403845248`), with Contents read, Pull requests write, and implicit Metadata
   read. The workflow requests only Contents read and Pull requests write and
   checks both installation and token repository scope. It needs no Contents
   write, administration, or workflow-write permission.
2. In repository Settings → Environments, create `kadan-pr-publishing`. Select
   **Selected branches and tags** and add exactly one **branch** rule: `master`.
   For the approved sole-owner design, do not require deployment reviewers;
   Andrew's explicit manual dispatch authorizes publication. Disable administrator
   bypass. Self-review prevention is optional GitHub functionality, not a
   requirement here: combined with required review it would prevent Andrew from
   approving his own dispatch. Any existing review rules still apply; this code
   does not remove them or change live settings.
3. Add `AI_KADAN_APP_PRIVATE_KEY` only as a secret of that environment, using the
   App's PEM private key. Do not add it as a repository or organization secret:
   those scopes can expose it to other workflows. The Personal Vault value in
   Codex is not automatically a GitHub environment secret. Never paste the key
   into inputs, source files, logs, or PR text. This PR does not create the
   environment, store a key, or change App permissions.
4. Keep existing branch protection and the requirement for approval of the most
   recent reviewable push. Bot PR authorship does **not** bypass that rule: if
   Andrew is the last pusher, a different eligible reviewer must approve.
   Do not weaken protection to make this workflow usable.

Kadan was public when this workflow was prepared. GitHub supports environment
secrets and protection rules for public repositories on current plans; legacy
plans may not support them. The publisher fails closed if visibility changes,
public environment configuration cannot be read, or the sole branch policy is
not `master`. It checks before entering the environment job (avoiding accidental
environment creation) and again before minting the token. The API does not reliably expose the
administrator-bypass setting; the owner must verify it in Settings. No broader
secret scope is a fallback. A concurrent administrator changing settings remains
a trust boundary.

The approved tradeoff is owner-authorized publishing without a second person's
approval. **Trusted `master` workflows referencing this environment can access
its secret; the environment does not isolate the key to this one workflow.**
Protect and review changes to all trusted workflows and the default branch.
This publisher verifies actor `ghalrym` and account ID `177494187`, event sender,
and the triggering actor on reruns. A managed connection acting as `ghalrym`
has the same identity as Andrew; GitHub cannot distinguish the human from that
connection. Dispatch through it requires Andrew's explicit authorization.
These publisher checks do not constrain other trusted workflows that reference
the environment. PR review and last-pusher requirements remain separate.

### Publish a branch

After the workflow is merged to `master`, open Actions → **Publish draft PR as
Ai Kadan** → Run workflow as `ghalrym`, selecting **master** as the workflow branch. Enter
an existing flat `codex/name` branch, its exact lowercase 40-character head SHA,
and the title and description. Branch code is never checked out or executed:
the publisher checks out only the trusted workflow commit on `master`. Inputs
are parsed from the event JSON as data, never interpolated into shell commands.

The workflow serializes publishers, checks all matching PRs (open, closed, and
merged), and reports an existing PR instead of opening another. A closed match
blocks reuse of that head/base pair; use a new branch for a genuinely new PR.
It creates drafts only, with no review, merge, or branch mutation. If a request
times out or the branch moves during creation, inspect the PR list and run logs
before deciding what to do next; the publisher never blindly retries creation.
GitHub may replace an older pending dispatch with a newer one; running jobs
are not canceled by later dispatches.

The short-lived token expires within one hour and the official action revokes
it during job cleanup. Cancellation or runner loss can delay cleanup until
expiry. Standard GitHub-hosted runners are used; for this public repository
these do not incur Actions minutes charges. No live publishing run is performed
as part of preparing the workflow. Unit tests use mocked API responses and no
credentials:

```sh
python3 -m unittest discover -s .github/scripts -p 'test_*.py' -v
```

Implementation references: [official token action](https://github.com/actions/create-github-app-token),
[environment protection and plan support](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments),
and [required reviews](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
