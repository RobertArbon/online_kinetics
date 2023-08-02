import click

from celerity.datasets.datasets import download_dataset


@click.group()
def cli():
    """
    Celerity - a package for online training of deep kinetic models of biomolecular simulations
    """
    pass


@cli.command()
@click.option('--data_dir', default='data/processed', type=click.Path(file_okay=False, dir_okay=True),
              help='Directory to download data to.')
@click.argument('dataset', default='alanine-dipeptide-distances', type=click.Choice([
    'alanine-dipeptide-positions',
    'alanine-dipeptide-distances',
    'alanine-dipeptide-dihedrals',
]))
def datasets(data_dir, dataset):
    download_dataset(data_dir, dataset)