import cohere

import nebuly

nebuly.init(api_key="ciao")

co = cohere.Client("0FQ41jtm4rw1jOF9hjGK7CQmqQGBLOjpKCA1tE24")
streaming_gens = co.chat(message="Hello world!", stream=True)
for i, token in enumerate(streaming_gens):
    print(f"Token {i}: {token}")
