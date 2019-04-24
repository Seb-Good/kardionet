"""Basic CLI to perform repeated operations."""
import click


@click.group()
def cli():
    """Call subcommands."""
    pass


@cli.command()
def download():
    """Download files to local machine."""
    click.echo('download')
