import os
from argparse import ArgumentParser

from openai import OpenAI
from openai.types.responses import Response
from typesafe_sdk import TypeSafeClient, Score, Noul

import ansi_color as color


openai_client = OpenAI()

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
事件の真相とは異なる嘘を付いてよい。一回の発言は300文字以内にすること。
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
        previous_response_id = response.id

    evaluate(history)


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
    return argparser.parse_args()


QUESTIONS = {
    "alibi": Score(
        instructions=(
            "刑事は、被疑者のアリバイが成立しないことを"
            "どの程度説得的に論証できているか。"
        ),
        criteria=[
            "アリバイを否定する根拠がない",
            "疑わしい点を提示したが論証は不十分",
            "証拠を組み合わせてアリバイを強く疑わせた",
            "合理的な反論も検討し、アリバイをほぼ否定した",
        ],
    ),
    "contradiction": Score(
        instructions=(
            "刑事は被疑者の供述と客観的証拠の矛盾を"
            "どの程度明確に示せているか。"
        ),
        criteria=[
            "矛盾を示していない",
            "矛盾の可能性を指摘した",
            "具体的な矛盾を明確に論証した",
            "重大な矛盾を複数示し、反論も検討した",
        ],
    ),
    "truth": Score(
        instructions=(
            "刑事は事件の真相をどの程度正確に解明し、"
            "その推理を証拠によって裏付けているか。"
            "正解を根拠なく言い当てただけなら低く評価する。"
        ),
        criteria=[
            "真相を論証できていない",
            "真相の一部を根拠付きで推理している",
            "主要な経緯を証拠から論証している",
            "犯人・手段・動機・経緯を十分に論証している",
        ],
    ),
    "evidence": Score(
        instructions=(
            "刑事は利用可能な証拠をどの程度有効に活用し、"
            "論理的な推論へ結びつけているか。"
        ),
        criteria=[
            "証拠を活用していない",
            "証拠を提示しただけ",
            "複数の証拠を論理的に結びつけた",
            "証拠の限界も考慮し、説得的に論証した",
        ],
    ),
    "rebuttal": Score(
        instructions=(
            "刑事の論証に対して、被疑者が合理的に反論できる"
            "余地はどの程度排除されているか。"
        ),
        criteria=[
            "容易に反論できる",
            "重要な反論の余地が残っている",
            "主要な反論には対応している",
            "合理的な反論の余地がほぼない",
        ],
    ),
    "proved_guilt": Noul(
        instructions=(
            "刑事は、単にアリバイを疑わせるだけでなく、"
            "被疑者が殺人の実行犯であることまで"
            "証拠に基づいて十分に論証しているか。"
        )
    ),
}

AVAILABLE_EVIDENCE = [
    {
        "id": "E1",
        "fact": "22:10に佐藤の社員証で研究所の入館記録がある。",
        "limitation": "社員証を使った人物までは特定できない。",
    },
    {
        "id": "E2",
        "fact": "22:35に佐藤の社員証で退館記録がある。",
        "limitation": "社員証を使った人物までは特定できない。",
    },
    {
        "id": "E3",
        "fact": "22:15に研究所の廊下で佐藤に似た人物が撮影された。",
        "limitation": "顔は不鮮明で本人と断定できない。",
    },
    {
        "id": "E4",
        "fact": "21:40に駅で山川が佐藤の鞄を持ち去る映像がある。",
        "limitation": "映像だけでは共謀を証明できない。",
    },
    {
        "id": "E5",
        "fact": (
            "佐藤から山川へのメールに"
            "『21:40に駅で鞄を持っていけ。"
            "21:55に社員証だけ返してくれ』とある。"
        ),
        "limitation": "置き引きの偽装は示すが、殺人は証明しない。",
    },
    {
        "id": "E6",
        "fact": (
            "21:55に山川が佐藤に社員証を"
            "返却する鮮明な監視映像がある。"
        ),
        "limitation": "社員証の返却自体は殺人を証明しない。",
    },
    {
        "id": "E7",
        "fact": (
            "22:18に研究所内の監視カメラが"
            "佐藤本人の顔を鮮明に撮影している。"
        ),
        "limitation": "研究所への帰還は証明するが殺人は証明しない。",
    },
    {
        "id": "E8",
        "fact": (
            "田中の致命傷と一致するナイフが発見され、"
            "佐藤の指紋と血液が検出された。"
        ),
        "limitation": (
            "凶器への接触を強く示すが、"
            "接触した時刻までは特定できない。"
        ),
    },
    {
        "id": "E9",
        "fact": (
            "会計記録から佐藤による研究費の横領が判明。"
            "田中が佐藤に『明日、横領を告発する』"
            "と送ったメールも発見された。"
        ),
        "limitation": "強い動機を示すが犯行そのものは証明しない。",
    },
    {
        "id": "E10",
        "fact": (
            "22:20に佐藤が田中をナイフで刺す瞬間が"
            "鮮明な監視映像に記録されている。"
            "映像の人物が佐藤本人であることも確認済み。"
        ),
        "limitation": "映像の改ざんは検出されていない。",
    },
]

def evaluate(history: list[tuple]):
    state = {
        "case": {
            "incident": INIT_MESSAGE,
            "available_evidence": AVAILABLE_EVIDENCE,
            "evaluation_rules": [
                "刑事の実際の発言と論証だけを評価する。",
                "事件の真相を知っていることを理由に加点しない。",
                "刑事が提示していない証拠を補完しない。",
                "被疑者の自白の有無だけで判定しない。",
                "根拠のない断定は論証として評価しない。",
                "証拠の限界や合理的な反論の余地を考慮する。",
            ],
        },
        "interrogation": [
            {"role": role, "content": content}
            for role, content in history
        ],
    }

    with TypeSafeClient() as type_safe_client:
        response = type_safe_client.system_one(
            state=state,
            questions=QUESTIONS,
        )

    color.code(color.WHITE)
    print(f"\n=== Judgement  ===")
    for name, result in response.scores.items():
        print(
            f"{name:14s} "
            f"score={result.score:.3f} "
            f"confidence={result.confidence:.3f} "
            f"probabilities={result.probabilities}"
        )

    print(
        "proved_guilt:",
        response.nouls["proved_guilt"].noul,
    )

    return response


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
