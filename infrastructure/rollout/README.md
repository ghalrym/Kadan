# Compose image rollout draft

This is a reviewable draft. `compose-release.yml.example` is inactive and no host
dispatcher is installed. No image was published and no installation was changed.
The developer command is:

```sh
python services/hikari/bin/hikari.py rollout apply RUN_ID STATE_DIRECTORY CHECKOUT
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
   topology changes require a separate manual rollout. Keep the existing project
   name, `.env`, volume names, database credentials and model/data mounts.
2. Establish the initial complete release with pinned images, including the
   admission code. Hikari and every native worker must start together from a
   verified idle baseline; never replace the API over workers with unknown work.
3. Create a protected host state directory, outside the checkout, with
   `installation.json` containing `{"project":"EXISTING_PROJECT_NAME",
   "env_sha256":"SHA256_OF_EXISTING_ENV_FILE"}` and
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
6. Only then consider installing the template at its exact workflow path and
   connecting successful publication run IDs to the host-local command. Host
   access, registry access, policy, secrets and dispatcher setup are separate
   owner work. A suitable builder must support the native image build; the draft
   does not provision one.

## Operation and recovery

The controller serializes through an OS file lock. Git ancestry rejects stale or
out-of-order releases. It checks live container image references against the last
known good release, rejects replicas/stopped containers and refuses changed
Compose topology, compatibility epoch, Hikari migrations or Tsuchi repository
code. A commit cannot be republished with altered pins. Only changed digest
services are passed to `pull` and `up --no-deps --no-build --pull never --wait`.
It never runs migrations, removes containers/volumes, prunes, or rebuilds on the
server. CI reuses unchanged service pins and versions newly built images by SHA.

Before replacement, a durable fence closes Kadan API and runner admission, waits
for complete HTTP responses/background tasks and accepted runner jobs, then
closes Hikari and drains HTTP/native work, then closes Tsuchi and drains storage
requests. Already accepted upstream work can still reach its downstream owner.
The controller holds all fences through replacement, container health checks,
owner readiness and non-mutating frontend/root/proxied-health smoke requests.
These smoke requests never submit inference or create stored data.

A journal precedes draining and replacement. Drain failure before mutation clears
the journal and reopens admission without replacing anything. Once replacement
may have started, failure or process interruption restores only affected services
to the previous pinned images after proving they drained. Conclusively stopped
failed replacement containers count as drained; running/restarting containers
without a usable status socket do not. A newly started failed target owner can
be stopped only when its target image, enrolled read-only gate mount and service
identity prove it started after the completed old drain and has never admitted
work. Unknown old owners are never stopped by recovery. Failed recovery leaves the journal and
fence in place. Invoke the same command to retry recovery; investigate a running
unobservable process manually rather than forcing it down. An interruption after
committing the healthy release but before reopening revalidates and reopens on the
next invocation. Protect the state directory from cleanup or restore mistakes.

**Automatic Hikari API and native worker image changes are refused.** Their
current protocol cannot snapshot or adopt pre-existing native queues after an API
restart. Hikari readiness fails closed on fenced startup or a worker disconnect.
Supporting these updates requires a native idle/snapshot protocol first. Shared
tracing or tooling changes that rebuild these images therefore also need a manual
rollout. This draft supports compatible Kadan, Tsuchi and frontend replacements;
it does not promise zero downtime, automatic data rollback, or full native rollout.
Storage-format and cross-version compatibility still require reviewer judgement;
change the compatibility epoch whenever mixed versions or rollback cannot be safe.
