from pathlib import Path
import yaml

import argparse
from pydantic import BaseModel


BASE_TYPES = (int, float, str)


class Arguments(argparse.ArgumentParser):
    def __init__(self, model: type[BaseModel], *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def parse(self):
        for name, field in self.model.model_fields.items():
            if field.json_schema_extra is None:
                continue
            aliases = field.json_schema_extra.get("aliases")
            if not aliases:
                continue
            if field.is_required():
                self.add_argument(
                    *aliases,
                    dest=name,
                    type=field.annotation if field.annotation in BASE_TYPES else str,
                    required=True,
                    help=field.description,
                )
            elif aliases:
                self.add_argument(
                    *aliases,
                    dest=name,
                    type=field.annotation if field.annotation in BASE_TYPES else str,
                    default=field.default,
                    help=field.description,
                )
        self.add_argument(
            "--config",
            dest="config",
            type=str,
            default=None,
            help="A file from which to load the configuration, if you do not "
            "want to use CLI arguments. If both are specified, values in config file will "
            "always take precedence.",
        )

        values = vars(self.parse_args())
        if config := values.pop("config", None):
            config = Path(config)
            if not config.is_file():
                raise FileNotFoundError(config)
            with config.open() as ifile:
                values.update(**yaml.safe_load(ifile).get(self.model.__name__, {}))
            if not values.get("output"):
                values["output"] = f"config={str(config).replace('/', '\\')}/data.hdf5"
        return self.model(**values)
