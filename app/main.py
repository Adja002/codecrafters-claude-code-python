import argparse
import json
import os
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 1. Initialize conversation history to persist across loop iterations
    messages = [{"role": "user", "content": args.p}]

    # 2. Define the available tools structure
    tools_schema = [
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

    # 3. Enter the Agent Loop
    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=tools_schema
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        assistant_message = chat.choices[0].message
        
        # Record the assistant's response in our tracking history
        messages.append(assistant_message)

        # 4. Check if the model requested any tool calls
        if assistant_message.tool_calls and len(assistant_message.tool_calls) > 0:
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name

                if function_name == "Read":
                    # Parse the target file parameters
                    arguments = json.loads(tool_call.function.arguments)
                    file_path = arguments.get("file_path")

                    # Execute tool internally (Do NOT print contents to stdout now!)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            tool_result = f.read()
                    except Exception as e:
                        tool_result = f"Error reading file: {str(e)}"

                    # Feed the result context back into our conversation array
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
        else:
            # 5. Break conditions: Model gave text without calling a tool
            if assistant_message.content:
                # Use standard print or write to deliver the final compiled answer
                sys.stdout.write(assistant_message.content)
                sys.stdout.flush()
            break


if __name__ == "__main__":
    main()
