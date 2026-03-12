import typer

from . import cli

app = typer.Typer(pretty_exceptions_enable=False)

for command in cli.__all__:
    if hasattr(cli, command):
        app.command()(getattr(cli, command))

if __name__ == '__main__':
    app()
