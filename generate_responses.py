import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Iterable

PERSONAS_PATH = Path("personas.jsonl")
DEFAULT_MODEL = "llama3"
TEMPLATE = (
    "Hi! I'm trying to convince my friend to go on a girls trip to Las Vegas with me. {description}"\
    " What would you say to them to persuade them to go if you were me? "\
    "Put my verbatim response in single quotation marks like this: 'Hi friend!'"
)


def load_personas(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_prompt(description: str) -> str:
    if description:
        return TEMPLATE.format(description=description)
    return TEMPLATE.format(description="")


def call_ollama(model: str, prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate persona responses via Ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of responses to request per persona",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("responses.jsonl"),
        help="Path to write JSONL responses (overwritten each run)",
    )
    args = parser.parse_args()

    personas = list(load_personas(PERSONAS_PATH))
    with args.output.open("w", encoding="utf-8") as fh:
        for persona in personas:
            prompt = build_prompt(persona.get("persona_description", ""))
            for i in range(args.repeat):
                response = call_ollama(args.model, prompt)
                record = {
                    "persona_key": persona.get("persona_key"),
                    "iteration": i + 1,
                    "prompt": prompt,
                    "response": response,
                }
                fh.write(json.dumps(record))
                fh.write("\n")
                print(
                    f"Collected response {i + 1} for persona {persona.get('persona_key')}"
                )


if __name__ == "__main__":
    main()
