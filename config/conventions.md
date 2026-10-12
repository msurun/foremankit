## Code conventions (clean code, readable by a beginner)

All code you write follows these rules. The reader may be a junior who has never seen this project: they must understand
what the code does and why without running it.

1. **Names reveal intent.** `remaining_budget_usd`, not `rb`. Functions are verbs (`load_nodes`), classes are nouns,
   booleans read as questions (`is_idle`). No abbreviations; one-letter names only for `i`, `j`, `k` in a tiny loop.
2. **Small functions with one job.** At most 20 lines of code (blank lines, comments and the docstring do not count), 4 parameters and 3 levels of nesting. One level of abstraction per
   function. When a block needs a comment to explain what it does, extract it into a function named after that comment.
3. **No duplication, no magic values.** A number or string with a meaning becomes a named constant, and the comment says
   its unit (`TIMEOUT_S = 30`).
4. **Errors are loud and helpful.** Raise a specific exception with a message that says what to do. Never a bare
   `except:`, never swallow an error silently, never return `None` or `-1` as an error code when an exception is clearer.
5. **Keep logic apart from side effects.** Pure functions for decisions and calculations; file, network and database
   access in thin separate functions. Pass values as arguments instead of reading global state.
6. **Comments explain WHY, docstrings explain WHAT.** Every module, class and public function has a docstring: a
   one-line summary, then what it returns or raises when that is not obvious. Comment the non-obvious: units, formulas,
   protocol quirks, the reason for a workaround. If a paragraph of logic is hard to follow, add an "in plain words" comment
   above it. Never comment what the line says (`i += 1  # add one`) and never leave commented-out code.
7. **Tests describe behaviour.** One behaviour per test; the name is a sentence (`test_a_node_that_is_off_is_skipped`);
   arrange, act, assert; no loops or conditions inside a test.
8. **Plain, boring Python.** PEP 8, lines up to 100 characters (as configured in ruff `line-length`; indentation counts), type hints on public functions, `pathlib`, f-strings.
   Data code (pandas): vectorised operations, and a comment with the columns and dtypes the code assumes.
9. **Every file starts with a header.** Two to four lines saying what the file does and why it exists: a module
   docstring in Python, a block of `//` comments in TypeScript, C and C++.
10. **Teach in numerical and domain code.** State the model, the formula (one line of derivation or the name of the
    method), the units and the data structure with its invariants where they are used. Write these comments together
    with the code, never in a later pass by a model that sees one function: such comments come out wrong or vacuous.
    When you review comments, look at the whole file and delete what only restates the code.
11. **When in doubt choose explicit over clever.** The shortest code is not the best code; the clearest is.

Do not reformat or rewrite code you were not asked to touch.
