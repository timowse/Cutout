# Contributing

Thanks for your interest in improving this project! Bug reports, test images,
quality comparisons, translations and code are all welcome.

## Principles

When in doubt, decide in this order:

**Privacy > simplicity > quality > speed > extra features**

- Images must never leave the user's device. No uploads, no analytics that see
  images or file names, no third-party requests.
- Keep the UI calm and focused: drop, paste or choose an image, get the result,
  download. No features that do nothing, no fake progress.
- Every dependency and every model must have a licence that allows free use,
  including commercial use (no "non-commercial" models, no AGPL components).

## Development setup

Requirements: Node.js ≥ 20.19 (22 recommended), Python ≥ 3.11 for the model
tool.

```bash
npm install
python3 -m pip install -r tools/model/requirements.txt
npm run model        # downloads and converts the AI model into public/models/ (once)
npm run dev          # http://localhost:5173
```

## Checks

```bash
npm run typecheck
npm run lint
npm test             # unit tests (Vitest)
npm run build
npx playwright test  # end-to-end tests against the build (uses a tiny mock model)
RUN_REAL_MODEL=1 npx playwright test tests/e2e/real-model.spec.ts   # real model, slow
```

`npm run check` runs typecheck, lint, unit tests and build in one go. Pull
requests must pass all of them (CI runs everything, including the real model).

## Quality reports

`npm run quality -- path/to/your/images` processes a folder of images with the
real model in headless Chromium and writes `quality-report/index.html`. This
runs entirely on your machine. Please share results (not private photos) when
proposing model or post-processing changes.

## Test images

Only add images you took yourself or that are clearly CC0 / public domain, and
note the source in `tests/fixtures/images/README.md`.

## Code style

- TypeScript strict mode, no `any`, no unexplained `eslint-disable`.
- Heavy work belongs in the worker, never on the UI thread.
- UI texts go into `src/i18n/en.ts` and `src/i18n/de.ts`.
- Keep comments for the *why*, not the *what*.

## Commit messages and pull requests

Describe what changed and why. For model or post-processing changes, include
before/after comparisons and measurements.

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
