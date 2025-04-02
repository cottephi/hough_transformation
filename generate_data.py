from src.argparser import Arguments
from src.datagenerator import DataGenerator
from src.plotter import Plotter
from src.objects import Deviations

from pathlib import Path

from pydantic import (
    ConfigDict,
    NonNegativeFloat,
    PositiveInt,
    Field,
    NonNegativeInt,
    BaseModel,
    field_validator,
)


class DataGeneratorArgs(BaseModel):
    __OUTPUT = Path("./generated_data")
    model_config = ConfigDict(extra="forbid")

    background_level: NonNegativeFloat = Field(
        0.01,
        description="Background noise as a fraction of the average signal in lines",
        aliases=["-b", "--background-noise"],
    )
    bins: tuple[int, int] = Field(
        "300x100",
        description="Number of pixels. If only an int is supplied, image dimensions are NxN, else NxM",
        aliases=["-B", "--bins"],
    )
    deviations: Deviations = Field(
        "0.2,0.0,0.5",
        description="The standard deviations of r, theta and of the Gaussian used"
        "to generate the signal across X and Y, as a tuple of 3 floats ",
        aliases=["-d", "--deviations"],
    )
    n_lines: PositiveInt = Field(
        3, description="Number of lines to generate", aliases=["-n", "--n_lines"]
    )
    outside_points: NonNegativeInt = Field(
        10,
        description="Number of points to generate outside of each line",
        aliases=["-o", "--outside-points"],
    )
    points_per_line: int = Field(
        50,
        min=2,
        description="Number of points to generate for each line",
        aliases=["-p", "--points-per-line"],
    )
    stddev: NonNegativeFloat = Field(
        0.2,
        description="Standard deviation of the signal intensity (mean signal intensity is always 1)",
        aliases=["-s", "--stddev"],
    )
    # DO NOT CHANGE THE ORDER OF 'output': it must be
    # after all other attributes in order to correctly determine the output path if not
    # specified
    output: Path = Field(
        "",
        description="The file name to save the generated data in. It will be located"
        " in ./generated_data/",
        aliases=["-O", "--output-path"],
    )

    @field_validator("bins", mode="before")
    def handle_bins(cls, bins: str | dict) -> tuple[int, int]:
        if isinstance(bins, str):
            return (
                tuple(int(value) for value in bins.split("x"))
                if "x" in bins
                else tuple(int(bins), int(bins))
            )
        return bins["x"], bins["y"]

    @field_validator("deviations", mode="before")
    def handle_deviations(cls, deviations: str | dict) -> Deviations:
        if isinstance(deviations, str):
            r, theta, spread = [
                float(value) for value in deviations.replace(" ", "").split(",")
            ]
            return Deviations(r=r, theta=theta, spread=spread)
        return Deviations(**deviations)

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
    parser = Arguments(
        model=DataGeneratorArgs,
        prog="DataGenerator",
        description="Program to generate lines for the Hough Transformation to find",
    )

    args = parser.parse()
    print("Using args", args)
    generator = DataGenerator(args)
    generator.generate()


if __name__ == "__main__":
    main()
