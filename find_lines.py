from src.argparser import Arguments
from src.linefinder import PointsFinder

from pathlib import Path

from pydantic import (
    Field,
    BaseModel,
    field_validator,
)


class LineFinderArgs(BaseModel):
    __OUTPUT = Path("./found_lines")

    threshold: float = Field(
        0.02,
        description="Threshold above which we consider data is signal",
        aliases=["-t", "--threshold"],
    )
    input: Path = Field(
        description="The HDF5 file containing the raw data",
        aliases=["-i", "--input"],
    )
    output: Path = Field(
        "",
        description="The file name to save the lines in. It will be located"
        " in ./found_lines/",
        aliases=["-o", "--output-path"],
    )

    @field_validator("input", mode="after")
    def handle_input(cls, path: Path) -> Path:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @field_validator("output", mode="before")
    def handle_output(cls, path: str, values) -> Path:
        if not path:
            path = (
                "_".join([f"{key}={value}" for key, value in values.data.items()])
                .replace(".", "")
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

    point_finder = PointsFinder(
        data=args.input, threshold=args.threshold, output=args.output
    )
    point_finder.find()


if __name__ == "__main__":
    main()
