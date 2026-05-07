# Security Policy

This is an experimental research repository. Please do not use it as a
production dependency without independent review.

## Reporting

If you find a vulnerability or a data exposure issue, please open a GitHub
security advisory or contact the maintainer through the email listed on the
GitHub profile.

Please avoid opening a public issue for suspected secrets, private data, or
exploit details until the maintainer has had time to respond.

## Scope

In scope:

- Secret exposure in repository files.
- Unsafe deserialization or command execution paths.
- Vulnerabilities in experiment scripts that could affect contributors.

Out of scope:

- Model quality failures.
- Theoretical weaknesses in proof claims unless they create a concrete software
  security issue.
- Vulnerabilities in local dependencies that are not exercised by this code.
