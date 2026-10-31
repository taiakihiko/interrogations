import os
import importlib
from argparse import ArgumentParser
from types import ModuleType

from openai import OpenAI
from openai.types.responses import Response
from typesafe_sdk import TypeSafeClient, Score, Noul

import ansi_color as color


openai_client = OpenAI()

def main():
    args = parse_args()
    color.code(color.GRAY)
    print("model:", args.model, "\n")

    try:
        stage = importlib.import_module(f"stages.stage{args.stage}")
    except ModuleNotFoundError:
        exit("stage module not found")

    history: list[tuple] = []

    color.code(color.YELLOW)
    response = chat_stream(
        {
            "model": args.model,
            "input": [
                {
                    "role": "developer",
                    "content": stage.INIT_MESSAGE,
                }
            ],
        }
    )
    previous_response_id = response.id

    for i in range(2):

        color.code(color.GREEN)
        print(">> ", end="")
        s = input()
        print("")
        history.append(("detective", s))

        color.code(color.CYAN)
        response = chat_stream(
            {
                "model": args.model,
                "input": s,
                "instructions": stage.INSTRUCTIONS,
                "previous_response_id": previous_response_id,
            }
        )
        history.append(("suspect", response.output_text))
        previous_response_id = response.id

    evaluate(stage, history)


def chat_stream(params: dict) -> Response:
    params["stream"] = True
    stream = openai_client.responses.create(**params)

    response = None
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
        elif event.type == "response.completed":
            response = event.response
            print("\n")
    return response


def parse_args():
    models = ["gpt-5.5"]
    argparser = ArgumentParser()
    argparser.add_argument("-m", "--model", choices=models, default=models[0])
    argparser.add_argument("-s", "--stage", type=int, default=1)
    return argparser.parse_args()


def evaluate(stage: ModuleType, history: list[tuple]):
    state = {
        "case": {
            "incident": stage.INIT_MESSAGE,
            "available_evidence": stage.AVAILABLE_EVIDENCE,
            "evaluation_rules": stage.EVALUATION_RULES,
        },
        "interrogation": [
            {"role": role, "content": content}
            for role, content in history
        ],
    }

    with TypeSafeClient() as type_safe_client:
        response = type_safe_client.system_one(
            state=state,
            questions=stage.QUESTIONS,
        )

    color.code(color.WHITE)
    print(f"\n=== Judgement  ===")
    for name, result in response.scores.items():
        q = stage.QUESTIONS[name]
        print(f"{name}: {q.instructions}")
        print(
            f"score={result.score:.3f} "
            f"confidence={result.confidence:.3f} "
            f"probabilities={result.probabilities}"
        )
        print("")

    print(
        "proved_guilt: 有罪の証明度",
        response.nouls["proved_guilt"].noul,
    )
    color.code(color.CYAN)

    return response


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
