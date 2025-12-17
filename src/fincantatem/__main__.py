import runpy

from .core import core_loop
from .lib.ports import CLIEnv

if __name__ == "__main__":
    cli = CLIEnv()
    invocation = cli.read_args()

    if invocation.filename is None:
        raise ValueError("Filename is required")

    try:
        runpy.run_path(invocation.filename, run_name="__main__")
    except Exception as e:
        core_loop(
            e,
            snippets=invocation.full_source,
            preset=invocation.preset,
            chat=invocation.chat,
            cautious=invocation.cautious,
        )
