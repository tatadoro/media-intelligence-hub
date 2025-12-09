import pathlib
import yaml


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_config(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config(CONFIG_PATH)

    print(f"Проект: {config.get('project_name')}")
    print("Старт каркаса проекта.")
    print("Пути к данным:")
    print(f"  raw:      {config['data']['raw_dir']}")
    print(f"  processed:{config['data']['processed_dir']}")
    # здесь позже:
    # - загрузка данных
    # - предобработка
    # - анализ / модели
    # - генерация отчётов из templates/


if __name__ == "__main__":
    main()
