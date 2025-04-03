from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource


class SubYamlConfigSettingsSource(YamlConfigSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings]):
        self.subfield = settings_cls.model_config["cli_prog_name"]
        YamlConfigSettingsSource.__init__(self, settings_cls)

    def _read_file(self, file_path):
        return super()._read_file(file_path).get(self.subfield, {})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", yaml_file="configs/config.yml")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        **_,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return SubYamlConfigSettingsSource(settings_cls), CliSettingsSource(
            settings_cls, cli_parse_args=True
        )