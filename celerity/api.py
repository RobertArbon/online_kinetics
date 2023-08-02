import click
from rich.pretty import pprint

from celerity.datasets.datasets import download_dataset
from celerity.models import estimator_by_type
from celerity.utils import dump_yaml, load_yaml


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
    """
    Download a dataset.
    data_dir: location to download the dataset to.
    dataset: a downloadable dataset.  See below for options.
    """
    download_dataset(data_dir, dataset)


@cli.command()
@click.argument('npz_dataset_path', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.argument('estimator_type',  type=click.Choice([
    'online',
    'batch'
]), default='batch')
def configure_training(npz_dataset_path, estimator_type):
    """
    Create a training configuration file.
    npz_dataset_path: path to a numpy npz dataset
    estimator_type: type of
    """
    params = estimator_by_type[estimator_type].get_options()
    params['npz_dataset_path'] = npz_dataset_path
    params['estimator_type'] = estimator_type
    dump_yaml(params, 'config.yaml')


@cli.command()
@click.argument('config_path', type=click.Path(file_okay=True, dir_okay=False, exists=True))
def train(config_path):
    params = load_yaml(config_path)
    pprint(params)
    est = estimator_by_type[params['estimator_type']]
    est.fit()



