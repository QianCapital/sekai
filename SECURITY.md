# Security Policy

## Supported versions

Security fixes are made for the latest minor release on PyPI.

| Version | Supported |
|---|---|
| 0.2.x | ✅ |
| < 0.2 | ❌ |

## Reporting a vulnerability

Please **do not open a public issue**. Report it privately through GitHub's
[private vulnerability reporting](https://github.com/QianCapital/sekai/security/advisories/new)
with a description, the affected versions and, if possible, a reproduction.

You can expect an acknowledgement within a few days. We will keep you informed while we
work on a fix and credit you in the release notes unless you prefer otherwise.

Note that environments, plugins and pickled objects loaded through sekai (for example
registry entry points, `register_namespace` loaders, or `ProcVecEnv` env factories) run
arbitrary Python code by design; only load ones you trust.
