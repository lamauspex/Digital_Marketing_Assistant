
""" Вспомогательные утилиты """


from pathlib import Path


def ensure_directory_exists(path):
    """Создаёт директорию, если её ещё нет"""
    directory = Path(path)
    if not directory.exists():
        directory.mkdir(
            parents=True,
            exist_ok=True
        )
