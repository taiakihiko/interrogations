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
    print("model:", args.model)

    try:
        stage = importlib.import_module(f"stages.stage{args.stage}")
    except ModuleNotFoundError:
        exit("stage module not found")

    color.code(color.WHITE)
    print(stage.COMMON_FACTS)

    color.code(color.YELLOW)
    print("--- 初期化応答 ---")
    response = chat_stream(
        {
            "model": args.model,
            "input": [
                {
                    "role": "developer",
                    "content": stage.COMMON_FACTS + stage.HIDDEN_TRUTH,
                }
            ],
        }
    )
    previous_response_id = response.id

    # TODO: プレイヤーの発話量に制限が必要。今は雑にターン数だけ制限。

    history: list[tuple] = []

    for i in range(args.turn):

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
                "instructions": stage.LLM_INSTRUCTIONS,
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
    argparser.add_argument("-t", "--turn", type=int, default=3)
    return argparser.parse_args()


def evaluate(stage: ModuleType, history: list[tuple]):
    state = {
        "case": {
            "incident": stage.COMMON_FACTS + stage.HIDDEN_TRUTH,
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
            questions=stage.JEV_QUESTIONS,
        )

    color.code(color.WHITE)
    print(f"\n=== Judgement  ===")
    for name, result in response.scores.items():
        question_name = stage.JEV_QUESTIONS_NAMES[name]
        socre_percent = result.score / (len(result.probabilities) - 1) * 100
        print(f"{question_name} ({name}): {socre_percent:.0f}%")
        print(
            f"score={result.score:.3f} "
            f"confidence={result.confidence:.3f} "
            f"probabilities={result.probabilities}"
        )
        print("")

    print("{} (proved_guilt): {}%".format(
        stage.JEV_QUESTIONS_NAMES["proved_guilt"],
        int(response.nouls["proved_guilt"].noul * 100)
    ))
    color.code(color.CYAN)

    return response


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
