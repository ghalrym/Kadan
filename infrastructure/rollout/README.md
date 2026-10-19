# Compose image rollout

The publication workflow is installed at `.github/workflows/compose-release.yml`.
It publishes immutable images and a release artifact from reviewed, passing master
commits, without access to the deployment server. The example remains a reference;
no host dispatcher is installed by this workflow. The developer command is:

```sh
python services/hikari/bin/hikari.py rollout apply RUN_ID STATE_DIRECTORY CHECKOUT
# Optional host-local dispatcher; no service unit or process is installed here:
python services/hikari/bin/hikari.py rollout watch STATE_DIRECTORY CHECKOUT
```

The run must be the successful `push` to `master` run of the exact
`.github/workflows/compose-release.yml` publication workflow in `ghalrym/Kadan`.
The command downloads its `compose-release` artifact, checks the source commit,
and accepts only digest references under `ghcr.io/ghalrym/kadan/`. It never accepts
an arbitrary Compose file, command, registry, or local release manifest. The
publication workflow checks that current master came from a merged PR with a
write-capable reviewer's approval of the PR head, no outstanding requested changes,
and successful exact-commit runs of all three existing CI workflows. The GitHub
repository/workflow and host-owned enrollment state are the trust boundary.

## Enrollment and activation

The owner must first inspect the template and complete a manual enrollment. This
is intentionally not automated: the controller cannot adopt an arbitrary running
installation or prove a native worker's initial queue state.

1. Use the canonical `compose.yaml` topology with exactly one instance of each
   managed service. Runner overrides, alternate Compose files, replicas and
   topology changes require a separate manual rollout. The owner must attest
   `detached-independent` supervision: containers run detached and no attached
   `compose up` session, abort-on-container-exit policy or whole-stack supervisor
   can stop/restart other services when one container exits. Docker inspection
   cannot prove that external supervisor policy, so unsupported supervision must
   be removed before enrollment. Keep the existing project
   name, `.env`, volume names, database credentials and model/data mounts.
2. Establish the initial complete release with pinned images, including the
   admission code. Hikari and every native worker must start together from a
   verified idle baseline; never replace the API over workers with unknown work.
3. Create a protected host state directory, outside the checkout, with
   `installation.json` containing `{"project":"EXISTING_PROJECT_NAME",
   "env_sha256":"SHA256_OF_EXISTING_ENV_FILE",
   "supervision":"detached-independent"}` and
   `state.json` containing `{"good": RELEASE_OBJECT, "pending": null}`. Keep this
   state durable and do not edit it while a controller holds its `lock` file.
   `RELEASE_OBJECT` follows the `Release` model in `bin/rollout.py`: source commit,
   SHA-256 of canonical `compose.yaml`, compatibility epoch `"1"`, and the complete
   service-to-image digest map. API and runner share the Kadan image.
4. Create `gate/fence.json` with `{"closed":[]}`. Generate an initial image
   overlay using `Compose.overlay(release)`, inspect it, then manually enroll
   the running containers with that overlay. It mounts the **directory** `gate`
   read-only at `/run/kadan/rollout` and sets `KADAN_ROLLOUT_SERVICE` on all four
   owners. They must have live status sockets. The canonical project and its
   `.env` remain authoritative; deployment commands explicitly supply both. The
   enrolled `.env` byte hash must remain unchanged. Ambient Compose and application
   variables are scrubbed; Docker uses the local `/var/run/docker.sock` only.
   Rootless/remote Docker endpoints require separate manual support.
5. Verify the initial running image pins, all seven worker connections, readiness,
   empty work queues and the existing storage volumes. Perform owner-operated
   manual request, streaming, runner, failure/restart and recovery checks on a
   disposable installation before enabling a real host dispatcher.
6. Only after verification start the host-local `watch` command or pass successful
   run IDs to `apply`. Host
   access, registry access, policy, secrets and dispatcher setup are separate
   owner work. A suitable builder must support the native image build; the draft
   does not provision one.

## Publication bootstrap before enrollment

Publication can run before any host enrollment so that genuine cached image pins
are available for disposable verification. A successful publication run does not
authorize a live rollout or start a dispatcher. The initial full build was checked
on a standard hosted CPU runner; no GPU execution or self-hosted builder is needed.

The activation PR must have a write-capable reviewer's GitHub approval of its exact
head before merge: the publisher checks that review and all three required
exact-master CI workflows. It uses only ephemeral `GITHUB_TOKEN` permissions
`contents: read`, `actions: read` and `packages: write`. No personal publication
token, host credential or repository permission change is required.

New GHCR packages default to private. The approved public bootstrap requires the
package owner to open each of the thirteen `ghalrym/kadan/*` package settings and
choose **Change visibility → Public**, confirming the package name. Public status
is irreversible. Do not broaden a CLI token if it lacks package-administration
access. If a pre-existing package is not linked to Kadan, its owner must also
grant this repository Actions access before publication can succeed.

Only after the approved visibility and caching steps should disposable recovery
verification run. Host enrollment and any unattended Actions credential handoff
remain independent, coordinated operations; this workflow supplies neither.

## Operation and recovery

The controller serializes through an OS file lock. Git ancestry rejects stale or
out-of-order releases. It checks live container image references against the last
known good release, rejects replicas/stopped containers and refuses changed
Compose topology, compatibility epoch, Hikari migrations or Tsuchi repository
code. A commit cannot be republished with altered pins. Changed digest services are passed to `pull` and
`up --no-deps --no-build --pull never --wait`. A Hikari API or native worker
change also restarts their coupled API-plus-seven-worker connection group;
unchanged images keep their existing digest pins and are not rebuilt.
It never runs migrations, removes containers/volumes, prunes, or rebuilds on the
server. CI reuses unchanged service pins and versions newly built images by SHA.

Before replacement, a durable fence closes Kadan API and runner admission, waits
for complete HTTP responses/background tasks and accepted runner jobs, then
closes Hikari and drains HTTP/native work, then closes Tsuchi and drains storage
requests. Already accepted upstream work can still reach its downstream owner.
Before fencing, the controller proves previous pins are cached locally, downloads
and verifies target pins, then restores the current overlay. Registry failure at
this stage does not close admission or stop containers. Replacement and recovery
use only cached pins with `--pull never`; recovery verifies its previous pins
locally before any stop and never depends on registry availability.

The controller holds all fences through replacement, container health checks,
owner readiness and non-mutating frontend/root/proxied-health smoke requests.
These smoke requests never submit inference or create stored data.

A journal precedes draining and replacement. Drain failure before mutation clears
the journal and reopens admission only after proving every recorded service process
epoch is unchanged, running and ready. Unknown stops/restarts preserve the fence
and journal for attention. Once replacement
may have started, failure or process interruption restores only affected services
to the previous pinned images after proving they drained. Conclusively stopped
failed replacement containers count as drained; running/restarting containers
without a usable status socket do not. A newly started failed target owner can
be stopped only when its target image, enrolled read-only gate mount and service
identity prove it started after the completed old drain and has never admitted
work. Unknown old owners are never stopped by recovery. A full service/container-ID
and process-start baseline is persisted before the first fence. After mutation
begins, recovery and pre-open verification require every **unaffected** epoch to
remain unchanged and running; a supervisor stop/restart is not hidden by a later
healthy status. The controller never restarts an unproven entire stack. Failed recovery leaves the journal and
fence in place. Invoke the same command to retry recovery; investigate a running
unobservable process manually rather than forcing it down. A durable `opening` receipt retains the resolved release, verified full process
baseline and runner eligibility evidence until admission has actually reopened.
Its baseline is captured before final health/lease verification and checked again
afterward. A closed-gate resume rechecks that exact baseline before and after
readiness/runner checks; failure preserves the receipt and fence without rolling
back the chosen release. If admission already opened before a crash, resume only
clears the receipt and never drains or rolls back newly accepted work. A closed
gate without an opening receipt requires manual investigation. Protect the state directory from cleanup or restore mistakes.

## Natural runner lease handoff

Tsuchi gives a runner process a 60-second infrastructure lease. A replacement uses
the same configured runner ID and a fresh worker UUID; the normal ownership check
rejects it while the old UUID's lease is live. The controller therefore records
the drained runner's exact container ID/start time, UUID, capacity, assigned-agent
count and existing lease expiry before stopping only that runner. Other services
stay running. It polls a read-only Tsuchi local status observation until that
**unchanged** lease naturally expires according to Tsuchi's own clock; it never
forces ownership transfer, rewrites assignment rows or bypasses HTTP admission.
Changed owner, renewed expiry, capacity/assignment drift, missing rows or missing
observations fail closed.

The fresh fenced runner exposes its configured ID, capacity and new process UUID.
Immediately before reopening, the controller proves that its UUID does not own
another runner, its capacity still fits the existing assignments and the old
recorded lease is expired (or the expected fresh UUID already legitimately owns
the row). This is a **registration eligibility** proof under the single-canonical-
runner enrollment assumption, not an observation of new-owner registration:
Tsuchi stays fenced and the ordinary registration PUT still runs after admission
reopens, preserving its existing 409 ownership checks. The same stop/expiry
handoff runs during recovery, including any known replacement lease. A failed
supervisor or unknown runner epoch retains the fence and journal for investigation.

## Proven-idle native replacement

Hikari API and all seven native workers form a coupled replacement group because
an API disconnect ends the native transport lifecycle. Before stopping any member,
the fixed local `barrier` command freezes internal maintenance/control submission
and proves a complete, stable connected cohort. Every native inference **and store**
lane must report `busy=false`, `queued=0` and `releasing=false`; API HTTP, job and
completion-task counts must also be zero. A disconnect, missing/duplicate worker,
unknown status, busy lane or changed process identity cannot authorize a stop.
Expected native cleanup may be polled within the bounded 20-second barrier budget;
unknown connection/status history fails immediately.
There is no snapshot, transfer or adoption of unfinished work.

The controller durably records this idle receipt, API/process generation UUIDs
and immutable Docker container IDs/start times, then revalidates the live receipt
before the first stop. It stops the recorded worker IDs before the API ID and
proves every persisted old container stopped or conclusively removed, including
IDs absent from Compose's current listing. Unknown/restarted/hidden old processes
keep the fence closed and require attention.

Only after the old group is stopped does the controller write a fresh cold-start
UUID marker to the read-only mounted gate directory. It recreates the pinned API
and all native workers with that transaction UUID, waits for a new complete idle
barrier and checks fresh API/native generations, Docker container IDs, hostnames,
image pins and marker binding. An old process cannot be accepted into the cold
cohort. The API stays fenced until this proof, health and smoke checks pass.
Protocol version 5 is required; manual enrollment must install it across the whole
group before automatic operation. Model files, downloads and database volumes stay
mounted with their existing identities.

Recovery repeats the same coupled cold start using the previous pinned release.
It may stop only the original proven-idle epochs or replacement epochs carrying a
recorded transaction UUID under the complete persistent fence. The journal survives
interruption before/after each stop/start boundary. A daemon error is never treated
as a missing old container. Unknown epochs or failed recovery preserve the journal
and fence. After a crash/reboot, unexpected automatic container restarts may change
an original epoch; that uncertainty requires manual investigation rather than a
forced deployment.

## Local polling and attention

`watch` is an explicit host-local process, inactive until the owner starts it.
It polls the fixed successful publication workflow every minute and shares the
same durable lock as `apply`. Pending recovery always runs before any network
lookup. Artifact provenance and Git source ancestry govern admission; workflow run
numbers do not stand in for commit order. A bounded receipt history suppresses
repeat deliveries, while late completions at lower run numbers are still examined.
A proven older commit is recorded and skipped without hiding another descendant.
The cursor advances only after verified stale/idempotent delivery or successful
application. The checkout is never switched; only commit objects are fetched from
the fixed repository for ancestry checks.

Read-only discovery failures use bounded subprocess timeouts and retry with a
one-to-five-minute backoff. Deployment, compatibility, proof or recovery failures
log the reason and terminate the dispatcher for operator attention; a bad image
is not redeployed endlessly after recovery. Restart the command explicitly after
investigation to retry the unconsumed candidate. KeyboardInterrupt/SIGTERM leave
the durable journal for the next invocation. Logs and nonzero exit are the supplied
attention signal; notification/supervision and all host activation/access setup
remain separate owner work. Reset the dispatcher receipt history when manually
enrolling a different last-known-good baseline.

Storage-format and cross-version compatibility still require reviewer judgement;
change the compatibility epoch whenever mixed versions or rollback cannot be safe.
The controller does not promise zero downtime or automatic data rollback. Its
automated tests mock external boundaries. Successful publication and compilation
do not prove deployed runtime health; coordinated disposable and enrollment
verification remain separate. Publication never runs inference or installs a host
service.
