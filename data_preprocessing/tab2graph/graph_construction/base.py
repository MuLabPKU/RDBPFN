import abc
from pathlib import Path

from dbinfer_bench import DBBRDBDataset

class GraphConstruction:

    config_class = None
    name = "base"

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def build(
        self,
        dataset : DBBRDBDataset,
        output_path : Path
    ):
        pass
