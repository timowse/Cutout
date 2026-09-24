# Security policy

## Reporting a vulnerability

Please report security issues **privately** via GitHub's
["Report a vulnerability"](https://github.com/timowse/Hintergrund-entfernen/security/advisories/new)
form (Security → Advisories), not in public issues.

Please include steps to reproduce and the browser/OS you used. You will get a
response as soon as possible; fixes are released through the normal deployment.

## Scope

Especially relevant are issues that could:

- send image data, file names or other user data anywhere (the app must never
  do this),
- bypass the Content Security Policy or load third-party code,
- execute code through crafted image files, or
- crash or freeze the browser with crafted inputs despite the size limits.

## Design notes

- The app is a static site without a backend; images are processed only in the
  browser (Web Worker + WebAssembly/WebGPU).
- The Content Security Policy restricts network access to the site's own
  origin (`connect-src 'self'`) and forbids inline scripts and `eval`
  (`'wasm-unsafe-eval'` is required to compile WebAssembly). The inference
  worker is started from a `blob:` bootstrap so it inherits this policy.
- File types are detected from file contents, dimensions are checked before
  decoding, and output sizes are capped per device.
- The AI model is downloaded from the same origin and each chunk's SHA-256 is
  verified against the manifest before use.

## Supported versions

Only the currently deployed version (the `main` branch) is supported.
