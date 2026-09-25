# CLAUDE.md

Project instructions for Claude Code working in this repository.

## Writing style

- Do not use em-dashes. Use a comma, a period, or a parenthesis instead.
- Do not use semicolons. Split into two sentences.
- Do not use bold or italic formatting in markdown, notebooks, or commit messages.
- Do not use LLM lingo. Avoid words and phrases like leverage, delve, seamless, robust, comprehensive, utilize, furthermore, streamline, cutting-edge, game-changer, elevate, unlock, unleash, dive into, at the end of the day, it's worth noting, boils down to.
- Write plainly. Say what something does, not how impressive it is.

## Git workflow

- Work in atomic commits. Each commit should contain one logical change and should make sense on its own.
- Do not bundle unrelated changes into a single commit.
- Before every commit, run:

  ```
  uv run ruff check .
  uv run ruff format --check .
  ```

- Fix anything ruff flags before committing. Never commit code that fails lint or formatting checks.

## Figures

- Invoke the `dataviz` skill before writing or changing any plotting code. Do not rely on memory of
  what it says, load it. It carries the form heuristic, the colour rules, the mark specifications
  and a runnable palette validator.
- Run the palette validator rather than judging colours by eye:

  ```
  node <dataviz skill path>/scripts/validate_palette.js "<hex,hex,...>" --mode light
  ```

- Figures should be readable by physicists, ML scientists, project managers,
  and directors. Favour large type, few marks, direct labels over legends where
  a legend would be hunted for, and one idea per figure.
- Look at every figure after rendering it. Open the PNG and check for label collisions, clipped
  marks, series hidden behind other series, and axis limits that exclude drawn data. The validator
  checks colour, not layout.
- Keep figure code in `src/mat_tt/report.py` so it is linted and tested, and register new figures in
  `FIGURES` so they export with the notebook figures.
