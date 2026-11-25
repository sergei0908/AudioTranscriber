import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    bot_token: str
    tmp_dir: str = "tmp"
    max_file_mb: int = 20

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024


def get_settings() -> Settings:
    token = os.getenv("TOKEN")
    if not token:
        raise RuntimeError("Не задан BOT_TOKEN")
    return Settings(
        bot_token=token,
        tmp_dir=os.getenv("TMP_DIR", "tmp"),
        max_file_mb=int(os.getenv("MAX_FILE_MB", "20")),
    )
