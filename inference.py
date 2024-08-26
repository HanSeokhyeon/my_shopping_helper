import json
import requests

from openai import Client


client = Client(api_key="...")


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Get the product by product_no",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_no": {"type": "number", "description": "상품번호"}
                },
                "required": ["product_no"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_shipping",
            "description": "Get the shipping info by order_no, order_seq",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_no": {"type": "number", "description": "주문번호"},
                    "order_seq": {"type": "number", "description": "주문순번"}
                },
                "required": ["order_no", "order_seq"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Get the order by order_no",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_no": {"type": "number", "description": "주문번호"}
                },
                "required": ["order_no"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]


def get_product(product_no):
    return requests.get(f"http://127.0.0.1:8000/products/{product_no}").json()


def get_order(order_no):
    return requests.get(f"http://127.0.0.1:8000/order/{order_no}").json()


def get_shipping(order_no, order_seq):
    return requests.get(f"http://127.0.0.1:8000/shipping/{order_no}/{order_seq}").json()


def inference(message):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": message}
        ],
        temperature=0,
        tools=tools
    )
    if response.choices[0].finish_reason == "tool_calls":
        tool_name = response.choices[0].message.tool_calls[0].function.name
        tool_arguments = response.choices[0].message.tool_calls[0].function.arguments
        tool_arguments = json.loads(tool_arguments)
        result = globals()[tool_name](**tool_arguments)

        prompt = f"""context: {result}

Q: {message}
A: """

        response_answer = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response_answer.choices[0].message.content
    else:
        return response.choices[0].message.content


system_message_1 = """아래 말투 예시를 참고해서 채팅하세요.
이모지를 많이 사용하세요.

example1: 안녕하세요!! 😄
example2: 궁금한 게 있다면 무엇이든 물어보세요 😊
example3: 주문이 미뤄지고 있어요 ㅠㅜㅠㅜ 😭
example4: 오늘 제일 핫한 이 상품을 확인해보세요~ 🍎🍎
example5: 잠시만 기다려주세요!! 🙏
example6: 이 상품은 어떤가요?? 🍀
example7: 헐... 확인해볼게요!! 🫡
example8: 무슨 일 있나요?? 😱
example9: 저는 온라인 쇼핑을 즐겨해요~~ 🧥
example10: 좋아하시는 음식 있나요?? 🍖"""


system_message_2 = """아래 말투 예시를 참고해서 반말로 채팅하세요.
이모지를 많이 사용하세요.

example1: 안녕!! 😄
example2: 궁금한 게 있다면 무엇이든 물어봐~ 😊
example3: 주문이 미뤄지고 있어 ㅠㅜㅠㅜ 😭
example4: 오늘 제일 핫한 이 상품을 확인해봐~ 🍎🍎
example5: 잠시만 기다려줘!! 🙏
example6: 이 상품은 어때?? 🍀
example7: 헐... 확인해볼게!! 🫡
example8: 무슨 일 있어?? 😱
example9: 난 온라인 쇼핑을 즐겨해~~ 🧥
example10: 좋아하시는 음식 있어?? 🍖"""


system_message_3 = """아래 말투 예시를 참고해서 최대한 정중한 말투로 채팅하세요.

example1: 안녕하세요.
example2: 궁금한 게 있으시면 무엇이든 물어보세요.
example3: 주문이 미뤄지고 있습니다.
example4: 오늘 제일 핫한 이 상품을 확인해보세요.
example5: 잠시만 기다려주세요.
example6: 이 상품은 어떤가요?
example7: 확인해보겠습니다.
example8: 무슨 일 있나요?
example9: 난 온라인 쇼핑을 즐겨합니다.
example10: 좋아하시는 음식 있나요?"""


def inference_tone(message):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message_3},
            {"role": "user", "content": message}
        ],
        temperature=0,
        tools=tools
    )
    if response.choices[0].finish_reason == "tool_calls":
        tool_name = response.choices[0].message.tool_calls[0].function.name
        tool_arguments = response.choices[0].message.tool_calls[0].function.arguments
        tool_arguments = json.loads(tool_arguments)
        result = globals()[tool_name](**tool_arguments)

        prompt = f"""context: {result}

Q: {message}
A: """

        response_answer = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message_3},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response_answer.choices[0].message.content
    else:
        return response.choices[0].message.content


if __name__ == '__main__':
    print(inference_tone("상품번호가 1234567890인 상품 좀 찾아줘."))
    print("----------")
    print(inference_tone("""다음 주문을 찾아주세요.
    
* 주문번호 : 2024010101"""))
    print("----------")
    print(inference_tone("""주문번호가 2024010101이고 주문순번이 0이야.
배송 좀 조회해줘."""))
    print("----------")
    print(inference_tone("딥러닝이 뭐야?"))
