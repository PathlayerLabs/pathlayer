# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it
privately instead: on the repository's **Security** tab, choose **Report a
vulnerability**. Only the maintainers see the report, and the fix can be
discussed and prepared there before anything is public.

Include what you can: the Pathlayer version or commit, the configuration, the
steps or trace that reproduce the problem, and what an attacker could do with
it.

## Supported versions

Pathlayer is alpha software. Fixes land on `main` and ship in the next
release; earlier releases are not patched.

## Scope

Pathlayer runs locally, next to the agent it observes. Reports we are
especially interested in:

- Data that Pathlayer stores (traces, run directories, the SQLite state,
  exported archives) readable beyond the permissions the documentation
  describes.
- Code executed through Pathlayer beyond what the operator configured.
- The OpenClaw ingest endpoint reachable from outside the host, while it is
  configured on its default loopback address.

The README documents trust boundaries that are not vulnerabilities on their
own: for example, an agent that can write to its own progress probe, its
tests or a configured `build.command` executable can forge progress or make
Pathlayer run its code.
