import argparse
from pydantic import BaseModel


BASE_TYPES = (int, float, str)


class Arguments(argparse.ArgumentParser):
    def __init__(self, model: type[BaseModel], *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def parse(self):
        for name, field in self.model.model_fields.items():
            if field.default is not None:
                default = field.default
            if field.default_factory is not None:
                default = field.default_factory()
            if default:
                self.add_argument(
                    *field.json_schema_extra["aliases"],
                    dest=name,
                    type=field.annotation if field.annotation in BASE_TYPES else str,
                    default=default,
                    help=field.description,
                )
            else:
                self.add_argument(
                    *field.json_schema_extra["aliases"],
                    dest=name,
                    type=field.annotation,
                    required=True,
                    help=field.description,
                )
        return self.model(**vars(self.parse_args()))
