import typer

import vit.cli

app = typer.Typer()

app.add_typer(vit.cli.app, name="vit")

if __name__ == "__main__":
    app()
