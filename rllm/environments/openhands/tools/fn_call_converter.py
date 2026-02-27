"""Utilities for adding tool schemas and examples in non-fn-calling mode."""

from __future__ import annotations


SYSTEM_PROMPT_SUFFIX_TEMPLATE = """
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
"""

IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX = """
--------------------- END OF NEW TASK DESCRIPTION ---------------------

PLEASE follow the format strictly! PLEASE EMIT ONE AND ONLY ONE FUNCTION CALL PER MESSAGE.
"""


def convert_tools_to_description(tools: list[dict]) -> str:
    """Convert tool schemas to a human-readable description block."""
    def _format_schema_details(
        schema: dict,
        indent: str = "      ",
    ) -> list[str]:
        """Recursively format nested schema details for object/array types."""
        lines: list[str] = []
        schema_type = schema.get("type")

        if schema_type == "object":
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))
            if properties:
                lines.append(f"{indent}Fields:")
                for child_name, child_info in properties.items():
                    child_type = child_info.get("type", "string")
                    child_status = (
                        "required" if child_name in required else "optional"
                    )
                    child_desc = child_info.get(
                        "description", "No description provided"
                    )
                    lines.append(
                        f"{indent}  - {child_name} ({child_type}, {child_status}): {child_desc}"
                    )
                    if "enum" in child_info:
                        enum_values = ", ".join(f"`{v}`" for v in child_info["enum"])
                        lines.append(
                            f"{indent}    Allowed values: [{enum_values}]"
                        )
                    lines.extend(_format_schema_details(child_info, indent + "    "))

        elif schema_type == "array":
            items = schema.get("items", {})
            if items and items.get("type", "any") == "object":
                lines.append(f"{indent}Items schema (object):")
                lines.extend(_format_schema_details(items, indent + "  "))

        return lines

    ret = ""
    for i, tool in enumerate(tools):
        if tool.get("type") != "function":
            continue
        fn = tool["function"]
        if i > 0:
            ret += "\n"
        ret += f"---- BEGIN FUNCTION #{i + 1}: {fn.get('name', 'unknown')} ----\n"
        ret += f"Description: {fn.get('description', 'No description provided')}\n"

        if "parameters" in fn:
            ret += "Parameters:\n"
            properties = fn["parameters"].get("properties", {})
            required_params = set(fn["parameters"].get("required", []))

            for j, (param_name, param_info) in enumerate(properties.items()):
                is_required = param_name in required_params
                param_status = "required" if is_required else "optional"
                param_type = param_info.get("type", "string")
                desc = param_info.get("description", "No description provided")

                if "enum" in param_info:
                    enum_values = ", ".join(f"`{v}`" for v in param_info["enum"])
                    desc += f"\n      Allowed values: [{enum_values}]"

                ret += (
                    f"  ({j + 1}) {param_name} ({param_type}, {param_status}): {desc}\n"
                )
                nested_lines = _format_schema_details(param_info, indent="      ")
                if nested_lines:
                    ret += "\n".join(nested_lines) + "\n"
        else:
            ret += "No parameters are required for this function.\n"

        ret += f"---- END FUNCTION #{i + 1} ----\n"
    return ret


def append_tools_to_system_prompt(system_prompt: str, tools: list[dict]) -> str:
    """Append tool schemas to system prompt for non-fn-calling models."""
    if not tools:
        return system_prompt
    if "You have access to the following functions:" in system_prompt:
        return system_prompt
    description = convert_tools_to_description(tools)
    suffix = SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(description=description)
    return f"{system_prompt}{suffix}"


def build_in_context_example(tools: list[dict]) -> str:
    """Build a compact in-context example using available tools."""
    available = {
        tool.get("function", {}).get("name")
        for tool in tools
        if tool.get("type") == "function"
    }
    available.discard(None)
    if not available:
        return ""

    example = """Here's a running example of how to perform a task with the provided tools.

--------------------- START OF EXAMPLE ---------------------

USER: Create a file named hello.txt and put "hello" in it.

ASSISTANT: Sure! Let me create the file:
"""
    if "str_replace_editor" in available:
        example += """<function=str_replace_editor>
<parameter=command>create</parameter>
<parameter=path>/workspace/hello.txt</parameter>
<parameter=file_text>hello</parameter>
</function>

USER: EXECUTION RESULT of [str_replace_editor]:
File created successfully at: /workspace/hello.txt

ASSISTANT: I will verify the file contents:
"""

    if "execute_bash" in available:
        example += """<function=execute_bash>
<parameter=command>
cat /workspace/hello.txt
</parameter>
</function>

USER: EXECUTION RESULT of [execute_bash]:
hello

ASSISTANT: The file contains the expected content.
"""

    if "finish" in available:
        example += """<function=finish>
<parameter=message>Created /workspace/hello.txt with content "hello".</parameter>
</function>
"""

    example += """
--------------------- END OF EXAMPLE ---------------------

Do NOT assume the environment is the same as in the example above.

--------------------- NEW TASK DESCRIPTION ---------------------
"""
    return example.lstrip()


def append_in_context_example(user_content: str, tools: list[dict]) -> str:
    """Inject an in-context example into the first user message."""
    example = build_in_context_example(tools)
    if not example:
        return user_content
    if user_content.startswith(example):
        return user_content
    if user_content.rstrip().endswith(IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX.strip()):
        return user_content
    return f"{example}{user_content}{IN_CONTEXT_LEARNING_EXAMPLE_SUFFIX}"


