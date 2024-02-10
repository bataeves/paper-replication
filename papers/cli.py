import typer
import papers.vit.cli

app = typer.Typer()

app.add_typer(papers.vit.cli.app, name="vit")

if __name__ == "__main__":
    app()
