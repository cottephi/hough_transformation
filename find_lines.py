from src.argparser.settings import Settings
from src.linefinder import LinesFinder

from pathlib import Path

from pydantic import (
    AliasChoices,
    Field,
    FilePath,
    field_validator,
)

from src.objects import RThetaBins, Spreads, Thresholds


class LineFinderArgs(Settings, cli_prog_name="LinesFinder"):
    __OUTPUT = Path("./found_lines")

    bins: RThetaBins = Field(
        RThetaBins(r=500, theta=500),
        validation_alias="bins",
    )
    input: FilePath = Field(
        description="The HDF5 file containing the raw data",
        validation_alias="input",
    )
    line_width: float = Field(
        1.0,
        description="Width of the line: points along a found line are considered"
        " 'on the line' if they are within this distance of the line",
        validation_alias=AliasChoices("line-width", "line_width"),
    )
    spreads: Spreads = Field(
        Spreads(xy=5, rtheta=15),
        validation_alias="spreads",
    )
    thresholds: Thresholds = Field(
        Thresholds(xy=1.0, rtheta=5.0),
        validation_alias="thresholds",
    )
    output: Path = Field(
        "",
        description="The file name to save the lines in. It will be located"
        " in ./found_lines/",
        validation_alias="output",
    )

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
    args = LineFinderArgs()
    print("Using args", args)

    lines_finder = LinesFinder(
        data=args.input,
        thresholds=args.thresholds,
        output=args.output,
        bins=args.bins,
        line_width=args.line_width,
        spreads=args.spreads,
    )
    lines_finder.find()


if __name__ == "__main__":
    main()
