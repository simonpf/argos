"""
argos.cli.main
==============

Main command line interface for argos.
"""
import click

from argos.cli.goes import goes
from argos.cli.seviri import seviri
from argos.cli.himawari import himawari
from argos.cli.gpm import gpm
from argos.cli.scampr import scampr
from argos.cli.mrms import mrms


@click.group()
@click.version_option(version="0.1.0")
def main():
    """ARGOS - Advanced Remote-sensing General Unified System.
    
    ML-based precipitation retrievals package for atmospheric science applications.
    """
    pass


main.add_command(goes)
main.add_command(seviri)
main.add_command(himawari)
main.add_command(gpm)
main.add_command(scampr)
main.add_command(mrms)


if __name__ == "__main__":
    main()
