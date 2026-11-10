import argparse
import json
import os
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p}],
        tools=[          
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read and return the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            }
        ]
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # Debug logs visible when running tests
    print("Logs from your program will appear here!", file=sys.stderr)

    # 1. Grab the assistant's message object
    message = chat.choices[0].message

    # 2. Check if the model requested any tool calls
    if message.tool_calls and len(message.tool_calls) > 0:
        first_tool_call = message.tool_calls[0]
        function_name = first_tool_call.function.name

        # 3. Handle the "Read" tool execution
        if function_name == "Read":
            # Parse the arguments string (which arrives as a JSON-formatted string)
            arguments = json.loads(first_tool_call.function.arguments)
            file_path = arguments.get("file_path")

            # Read the target file content and write it cleanly to standard output
            with open(file_path, "r", encoding="utf-8") as f:
                file_contents = f.read()
            
            # Use sys.stdout.write to ensure no extra whitespace or trailing newlines are forced
            sys.stdout.write(file_contents)
            sys.stdout.flush()
    else:
        # 4. Fallback: If no tool was called, print the standard textual response
        if message.content:
            print(message.content)


if __name__ == "__main__":
    main()
