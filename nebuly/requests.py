import json
import urllib.request


def post_json_data(url: str, data_dict: dict[str, str]) -> urllib.request.Request:
    json_data = json.dumps(data_dict).encode("utf-8")

    request = urllib.request.Request(
        url, data=json_data, headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=3) as response:
        return response.readlines()
