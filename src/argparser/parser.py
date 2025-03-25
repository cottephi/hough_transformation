import argparse
from pydantic import BaseModel


BASE_TYPES = (int, float, str)


class Arguments(argparse.ArgumentParser):
    def __init__(self, model: type[BaseModel], *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def parse(self):
        for name, field in self.model.model_fields.items():
            if field.is_required():
                self.add_argument(
                    *field.json_schema_extra["aliases"],
                    dest=name,
                    type=field.annotation if field.annotation in BASE_TYPES else str,
                    required=True,
                    help=field.description,
                )
            else:
                self.add_argument(
                    *field.json_schema_extra["aliases"],
                    dest=name,
                    type=field.annotation if field.annotation in BASE_TYPES else str,
                    default=field.default,
                    help=field.description,
                )
        return self.model(**vars(self.parse_args()))
