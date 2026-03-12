import typer
import dbinfer_bench as dbb

def list_builtin():
    print('\n'.join(dbb.list_builtin()))

def download(
    dataset_name : str = typer.Argument(
        ...,
        help="Dataset name to download."
    ),
    version : str = typer.Option(
        None,
        help="Dataset version."
    ),
):
    dataset_path = dbb.get_builtin_path_or_download(dataset_name, version)
    print(f"Dataset downloaded to '{dataset_path}'.")
