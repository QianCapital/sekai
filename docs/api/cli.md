# CLI

The `sekai` command (also `python -m sekai`). Usage is documented in the
[Command line guide](../guide/cli.md); this page covers the programmatic entry points, which
are handy for embedding the commands in your own tooling or tests.

```python
from sekai.cli import main

exit_code = main(["check", "CartPole-v1", "--steps", "100"])  # prints "CartPole-v1: all checks passed."
print(exit_code)  # 0
```

::: sekai.cli
    options:
      show_root_heading: false
      members:
        - main
        - build_parser
      show_if_no_docstring: true
