from src.argparser import Arguments
from src.datagenerator import DataGenerator
from src.plotter import Plotter
from src.objects import Deviations

from pathlib import Path

from pydantic import (
    NonNegativeFloat,
    PositiveInt,
    Field,
    NonNegativeInt,
    BaseModel,
    field_validator,
)


class DataGeneratorArgs(BaseModel):

    __OUTPUT = Path("./generated_data")

    n_lines: PositiveInt = Field(
        4, description="Number of lines to generate", aliases=["-n", "--n_lines"]
    )
    deviations: Deviations = Field(
        "0.2,0.01,0.2",
        description="The standard deviations of r, theta and of the Gaussian used"
        "to generate the signal across X and Y, as a tuple of 3 floats ",
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
    bins: tuple[int, int] = Field(
        "300x100",
        description="Number of pixels. If only an int is supplied, image dimensions are NxN, else NxM",
        aliases=["-B", "--bins"],
    )
    stddev: NonNegativeFloat = Field(
        0.2,
        description="Standard deviation of the signal intensity (mean signal intensity is always 1)",
        aliases=["-s", "--stddev"],
    )
    output: Path = Field(
        "",
        description="The file name to save the generated data in. It will be located"
                    " in ./generated_data/",
        aliases=["-O", "--output-path"],
    )

    @field_validator("bins", mode="before")
    def handle_bins(cls, bins: str) -> tuple[int, int]:
        return (
            tuple(int(value) for value in bins.split("x"))
            if "x" in bins
            else tuple(int(bins), int(bins))
        )

    @field_validator("deviations", mode="before")
    def handle_deviations(cls, deviations: str) -> Deviations:
        r, theta, spread = [
            float(value) for value in deviations.replace(" ", "").split(",")
        ]
        return Deviations(r=r, theta=theta, spread=spread)

    @field_validator("output", mode="before")
    def handle_output(cls, path: str, values) -> Path:
        if not path:
            path = "_".join(
                [f"{key}:{value}" for key, value in values.data.items()]
            ).replace(".", "").replace(" ", "")
            path = cls.__OUTPUT.default / path / "data.hdf5"
        else:
            path  = cls.__OUTPUT.default / path
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
    image, lines = generator.generate()
    plotter = Plotter(image, args.bins, lines)
    plotter.plot(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
