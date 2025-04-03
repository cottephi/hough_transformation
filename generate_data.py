from src.argparser.settings import Settings
from src.datagenerator import DataGenerator
from src.objects import Deviations, XYBins

from pathlib import Path

from pydantic import (
    AliasChoices,
    NonNegativeFloat,
    PositiveInt,
    Field,
    NonNegativeInt,
    field_validator,
)


class DataGeneratorArgs(Settings, cli_prog_name="DataGenerator"):
    __OUTPUT = Path("./generated_data")

    background_level: NonNegativeFloat = Field(
        0.01,
        description="Background noise as a fraction of the average signal in lines",
        validation_alias=AliasChoices("background-level", "background_level"),
    )
    bins: XYBins = Field(
        XYBins(x=300, y=100),
        validation_alias="bins",
    )
    deviations: Deviations = Field(
        Deviations(r=0.2, theta=0.0, spread=0.5, signal=0.2),
        validation_alias="deviations",
    )
    n_lines: PositiveInt = Field(
        3,
        description="Number of lines to generate",
        validation_alias=AliasChoices("n-lines", "n_lines"),
    )
    outside_points: NonNegativeInt = Field(
        10,
        description="Number of points to generate outside of each line",
        validation_alias=AliasChoices("outside-points", "outside_points"),
    )
    points_per_line: int = Field(
        50,
        min=2,
        description="Number of points to generate for each line",
        validation_alias=AliasChoices("points-per-line", "points_per_line"),
    )
    output: Path = Field(
        "",
        description="The file name to save the generated data in. It will be located"
        " in ./generated_data/",
        validation_alias="output",
    )

    @field_validator("output", mode="before")
    def handle_output(cls, path: str, values) -> Path:
        if not path:
            path = (
                "_".join([f"{key}={value}" for key, value in values.data.items()])
                .replace(".", "")
                .replace(" ", "")
            )
            path = cls.__OUTPUT.default / path / "data.hdf5"
        else:
            path = cls.__OUTPUT.default / path
            if path.is_dir():
                path = path / "data.hdf5"
            if not path.suffix == ".hdf5":
                path = path.with_suffix(".hdf5")

        if not path.parent.is_dir():
            path.parent.mkdir(parents=True)
        return path


def main() -> None:
    args = DataGeneratorArgs()
    print("Using args", args)
    generator = DataGenerator(args)
    generator.generate()


if __name__ == "__main__":
    main()
