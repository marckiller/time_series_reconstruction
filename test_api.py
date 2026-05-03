import json

import requests


URL = "http://127.0.0.1:8000/reconstruct"


def main():
    with open("example.json", "r") as f:
        payload = json.load(f)

    response = requests.post(URL, json=payload, timeout=10)
    response.raise_for_status()
    body = response.json()

    reconstructed = body["reconstructed"]
    assert len(reconstructed) == 60
    assert abs(reconstructed[0] - payload["target_ohlc"]["open"]) < 1e-5
    assert abs(reconstructed[-1] - payload["target_ohlc"]["close"]) < 1e-5
    print(body)


if __name__ == "__main__":
    main()
