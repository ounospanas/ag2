# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from logging import getLogger
from typing import Annotated, Optional

import typer

from .. import __version__

app = typer.Typer(rich_markup_mode="rich")

logger = getLogger(__name__)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__version__}")
        raise typer.Exit()


@app.callback()
def callback(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", help="Show the version and exit.", callback=version_callback),
    ] = None,
) -> None:
    """AG2 mcp proxy CLI - The [bold]mcp proxy[/bold] command line app. 😎

    Generate mcp proxy for your [bold]AG2[/bold] projects.

    Read more in the docs: ...
    """  # noqa: D415e


@app.command()
def create() -> None:
    typer.echo(
        "This command is not available yet.",
        err=True,
    )


if __name__ == "__main__":
    app(prog_name="mcp_proxy")
