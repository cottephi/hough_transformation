from src.argparser import Arguments
from src.datagenerator import DataGenerator
from src.plotter import Plotter

from pathlib import Path

from pydantic import (
    NonNegativeFloat,
    PositiveInt,
    Field,
    NonNegativeInt,
    BaseModel,
    field_validator,
)


class Deviations(BaseModel):
    r: NonNegativeFloat
    theta: NonNegativeFloat


class DataGeneratorArgs(BaseModel):
    n_lines: PositiveInt = Field(
        4, description="Number of lines to generate", aliases=["-n", "--n_lines"]
    )
    deviations: Deviations = Field(
        "0.01,0.01",
        description="The standard deviations of r and theta as a tuple of floats "
        "(note that 0 <= r < 1 and 0 <= theta < pi)",
        aliases=["-d", "--deviations"],
    )
    points_per_line: int = Field(
        250,
        min=2,
        description="Number of points to generate for each line",
        aliases=["-p", "--points-per-line"],
    )
    outside_points: NonNegativeInt = Field(
        20,
        description="Number of points to generate outside of each line",
        aliases=["-o", "--outside-points"],
    )
    background_level: NonNegativeFloat = Field(
        0.1,
        descroption="Background noise as a fraction of the average signal in lines",
        aliases=["-b", "--background-noise"],
    )
    bins: int = Field(
        100,
        description="Number of pixels (image is a square)",
        aliases=["-B", "--bins"],
    )
    output: Path = Field(
        "./data.hdf5",
        description="The file to save the generated data in",
        aliases=["-O", "--output-path"],
    )
    stddev: NonNegativeFloat = Field(
        0.2,
        description="Standard deviation of the signal intensity (mean signal intensity is always 1)",
        aliases=["-s", "--stddev"],
    )

    @field_validator("deviations", mode="before")
    def handle_deviations(cls, deviations: str) -> Deviations:
        r, theta = [float(value) for value in deviations.replace(" ", "").split(",")]
        return Deviations(r=r, theta=theta)


def main() -> None:
    parser = Arguments(
        model=DataGeneratorArgs,
        prog="DataGenerator",
        description="Program to generate lines for the Hough Transformation to find",
    )

    args = parser.parse()
    print("Using args", args)

    generator = DataGenerator(args)
    image, lines = generator.generate()
    limits = (tuple(DataGenerator.X_Y_RANGE), tuple(DataGenerator.X_Y_RANGE))
    plotter = Plotter(image, limits, lines)
    plotter.plot(
        args.output.with_suffix(".pdf"),
    )


if __name__ == "__main__":
    main()
