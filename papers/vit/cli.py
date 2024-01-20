import typer

app = typer.Typer()


@app.command()
def train():
    pass


@app.command()
def demo():
    pass


if __name__ == "__main__":
    app()
