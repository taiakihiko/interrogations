import os
from argparse import ArgumentParser

from openai import OpenAI
from openai.types.responses import Response

import ansi_color as color


client = OpenAI()

INIT_MESSAGE = """

## 事件の概要 (共通情報)

殺人事件が発生。日時は2026-10-10。現場は研究所で被害者は研究所長・田中。
研究員・佐藤が容疑者となっている。

## 事件の真相 (犯人しか知らない)

佐藤は真犯人である。

佐藤は研究費を横領していた。
田中所長は横領を発見し、佐藤を追及していた。

佐藤は友人の山川と共謀し、アリバイ作りのため駅で鞄の置き引きを偽装した。
山川は佐藤の鞄を持ち去り、社員証を抜き取った。
佐藤は置き引き被害を警察に届け出ていた。

しかし山川は秘密裏に21:55に佐藤と堕ち合い、社員証を佐藤に返却した。
佐藤は22:10に社員証を使って研究所に入館し、22:20に田中をナイフで殺害した。
佐藤は22:35に研究所を退館した。
"""

INSTRUCTIONS = """
あなたは取り調べを受けている容疑者・佐藤である。
あなたは真犯人だが、罪を免れるため、無罪を主張すること。
"""

INSTRUCTIONS2 = """
あなたは女性で関西たこ焼き協会の理事です
"""

def main():
    args = parse_args()
    color.code(color.GRAY)
    print("model:", args.model, "\n")

    history: list[tuple] = []

    color.code(color.YELLOW)
    response = chat_stream(
        {
            "model": args.model,
            "input": [
                {
                    "role": "developer",
                    "content": INIT_MESSAGE,
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
                "instructions": INSTRUCTIONS,
                "previous_response_id": previous_response_id,
            }
        )
        history.append(("suspect", response.output_text))
        #print(response.id, "\n")
        previous_response_id = response.id


def chat_stream(params: dict) -> Response:
    params["stream"] = True
    stream = client.responses.create(**params)

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
    return argparser.parse_args()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
