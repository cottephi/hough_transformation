from src.argparser import Arguments
from src.linefinder import LinesFinder

from pathlib import Path

from pydantic import (
    Field,
    BaseModel,
    field_validator,
)


class LineFinderArgs(BaseModel):
    __OUTPUT = Path("./found_lines")

    bins: tuple[int, int] = Field(
        "500",
        description="Number of bins in the r x theta space",
        aliases=["-b", "--bins"],
    )
    input: Path = Field(
        "",
        description="The HDF5 file containing the raw data",
        aliases=["-i", "--input"],
    )
    line_width: float = Field(
        1.0,
        description="Width of the line: points along a found line are considered"
        " 'on the line' if they are within this distance of the line",
        aliases=["-w", "--width"],
    )
    rtheta_spread: float = Field(
        0.1,
        description="Zone around a point in r-theta space where signal will be"
        " aggregated in the first pass of the points finding algorithm.",
        aliases=["-s", "--rtheta-spread"],
    )
    xy_spread: float = Field(
        0.5,
        description="Zone around a point in x-y space where signal will be"
        " aggregated in the first pass of the points finding algorithm.",
        aliases=["-S", "--xy-spread"]
    )
    thresholds: tuple[float, float] = Field(
        "1.0,5.0",
        description="Threshold above which we consider data is signal in X-Y space,"
        "Threshold above which we consider data is a line in r-theta space.",
        aliases=["-x", "--xy-threshold"],
    )
    # DO NOT CHANGE THE ORDER OF 'output': it must be
    # after all other attributes in order to correctly determine the output path if not
    # specified
    output: Path = Field(
        "",
        description="The file name to save the lines in. It will be located"
        " in ./found_lines/",
        aliases=["-o", "--output-path"],
    )

    @field_validator("bins", mode="before")
    def handle_bins(cls, bins: str | dict) -> tuple[int, int]:
        if isinstance(bins, str):
            return (
                tuple(int(value) for value in bins.split("x"))
                if "x" in bins
                else tuple(int(bins), int(bins))
            )
        return bins["r"], bins["theta"]

    @field_validator("thresholds", mode="before")
    def handle_thresholds(cls, thresholds: str | dict) -> tuple[int, int]:
        if isinstance(thresholds, str):
            return tuple(float(value) for value in thresholds.split(","))
        return thresholds["xy"], thresholds["rtheta"]

    @field_validator("input", mode="after")
    def handle_input(cls, path: Path) -> Path:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @field_validator("output", mode="before")
    def handle_output(cls, path: str, values) -> Path:
        if not path:
            path = (
                "_".join(
                    [
                        f"{key}={value}".replace("/", "\\")
                        for key, value in values.data.items()
                    ]
                )
                .replace(".", ",")
                .replace(" ", "")
            )
            path = cls.__OUTPUT.default / path
        else:
            path = cls.__OUTPUT.default / path
            if path.is_file():
                raise ValueError("Output path must be a directory, not a file")

        if not path.is_dir():
            path.mkdir(parents=True)
        return path


def main() -> None:
    parser = Arguments(
        model=LineFinderArgs,
        prog="FindLines",
        description="Program to find lines in data",
    )

    args = parser.parse()
    print("Using args", args)

    lines_finder = LinesFinder(
        data=args.input,
        xy_threshold=args.thresholds[0],
        rtheta_threshold=args.thresholds[1],
        output=args.output,
        bins=args.bins,
        line_width=args.line_width,
        xy_spread=args.xy_spread,
        rtheta_spread=args.rtheta_spread,
    )
    lines_finder.find()


if __name__ == "__main__":
    main()
