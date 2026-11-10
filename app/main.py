import argparse
import json
import os
import subprocess
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

    # 2. Define the available tools (Read + Write + Bash schemas)
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
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path", "content"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path of the file to write to"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute"
                        }
                    }
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
                arguments = json.loads(tool_call.function.arguments)

                if function_name == "Read":
                    file_path = arguments.get("file_path")
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            tool_result = f.read()
                    except Exception as e:
                        tool_result = f"Error reading file: {str(e)}"

                elif function_name == "Write":
                    file_path = arguments.get("file_path")
                    content = arguments.get("content", "")
                    
                    dir_name = os.path.dirname(file_path)
                    if dir_name and not os.path.exists(dir_name):
                        os.makedirs(dir_name, exist_ok=True)
                        
                    try:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        tool_result = "File written successfully."
                    except Exception as e:
                        tool_result = f"Error writing file: {str(e)}"

                elif function_name == "Bash":
                    command = arguments.get("command")
                    try:
                        # Run the shell command inside the current active working directory
                        result = subprocess.run(
                            command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            check=False
                        )
                        # Combine stdout and stderr outputs so the model sees all outcomes or errors
                        tool_result = result.stdout + result.stderr
                    except Exception as e:
                        tool_result = f"Execution error: {str(e)}"
                else:
                    tool_result = f"Unknown tool: {function_name}"

                # Feed the result context back into our conversation array
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
        else:
            # 5. Break conditions: Model gave text without calling a tool
            if assistant_message.content:
                sys.stdout.write(assistant_message.content)
                sys.stdout.flush()
            break


if __name__ == "__main__":
    main()
