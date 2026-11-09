# Security policy

## Supported versions

Only the latest commit on `main` is supported. Fixes are not backported to older commits or
releases.

## Reporting a vulnerability

Report vulnerabilities privately to [ext-oss@airoa.org](mailto:ext-oss@airoa.org), or with
**Report a vulnerability** on the
[Security tab of airoa-org/yubi-real2sim](https://github.com/airoa-org/yubi-real2sim/security).
Do not open a public issue or pull request for it.

Include the affected commit, the steps to reproduce, and the impact you expect.

## Scope

This policy covers the code in this repository. Vulnerabilities in third-party components
(see [`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES)), such as OpenPI, robosuite or MuJoCo,
belong with their upstream projects. The downloaded checkpoints and scenes
are inputs that the release pins by SHA-256; report a published input whose content does not
match its pin through the same private channel.
