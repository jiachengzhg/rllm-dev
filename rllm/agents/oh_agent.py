"""
OpenHands-compatible Agent for Code Agent Training.

This agent implements OpenHands-style tool calling with support for:
- execute_bash
- str_replace_editor
- think
- finish
- task_tracker

This agent also supports token accumulation to avoid retokenization issues during training.
When tokenizer and chat_parser are provided, the agent maintains token sequences directly
instead of re-encoding messages at each step.
"""

import json
import logging
import re
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.environments.openhands.tools import get_all_tool_schemas
from rllm.environments.openhands.tools.fn_call_converter import (
    append_in_context_example,
    append_tools_to_system_prompt,
)

# Type hints for optional dependencies
try:
    from transformers import PreTrainedTokenizerBase
    from rllm.parser import ChatTemplateParser
except ImportError:
    PreTrainedTokenizerBase = None
    ChatTemplateParser = None

TOKEN_WARNING_THRESHOLD = 28000

logger = logging.getLogger(__name__)


# OpenHands-style system prompt
OPENHANDS_SYSTEM_PROMPT = """You are a software engineer agent tasked with solving GitHub issues and coding tasks.
"""

# Debug prompts for simple tool-call verification
SYSTEM_PROMPT_DEBUG = "You are an assistant who strictly executes commands according to user requests."

USER_PROMPT_DEBUG = """Please strictly output the following content first:
I'll create a hello_world.txt file under /testbed directory.
<function=str_replace_editor>
<parameter=command>create</parameter>
<parameter=path>/testbed/hello_world.txt</parameter>
<parameter=file_text>Hello, World!</parameter>
</function>
Then stop. You will get a response from the environment like this:
File created successfully at: /testbed/hello_world.txt
After you get the response, output the following content:
<function=finish>
<parameter=message>done</parameter>
</function>
"""


OPENHANDS_USER_PROMPT = """Consider the following github issue:
<github_issue>
{problem_statement}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I've already taken care of all changes to any of the test files described in the <github_issue>. 
This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <github_issue> is satisfied.

Follow these steps:
1. Explore the repo to familiarize yourself with its structure
2. Create a script ('reproduce_issue.py') to reproduce the error
3. Edit the sourcecode to resolve the issue
4. Rerun your reproduce script to confirm the fix
5. Think about edge cases and handle them

IMPORTANT: Each response must include both reasoning and a function call.
"""


def parse_xml_function_call(response_text: str) -> tuple[str, dict]:
    """
    Parse XML-style function call from response.
    
    Format: <function=name><parameter=key>value</parameter></function>
    
    Returns:
        Tuple of (function_name, arguments_dict)
    """
    # Find function name
    fn_match = re.search(r"<function=(\w+)>", response_text)
    if not fn_match:
        return "", {}
    
    function_name = fn_match.group(1)
    
    # Extract parameters
    arguments = {}
    # Match parameters, handling multiline values
    param_pattern = r"<parameter=(\w+)>(.*?)</parameter>"
    for match in re.finditer(param_pattern, response_text, re.DOTALL):
        param_name = match.group(1)
        param_value = match.group(2).strip()
        
        # Try to parse as JSON for complex types
        if param_value.startswith("[") or param_value.startswith("{"):
            try:
                param_value = json.loads(param_value)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {param_value}\nparam_name: {param_name}\nparam_value: {param_value}")
                pass
        
        arguments[param_name] = param_value
    
    return function_name, arguments


def format_action_xml(function_name: str, arguments: dict) -> str:
    """
    Format action as XML string.
    
    Args:
        function_name: Name of the function
        arguments: Dictionary of arguments
        
    Returns:
        XML formatted action string
    """
    parts = [f"<function={function_name}>"]
    for key, value in arguments.items():
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value)
        else:
            value_str = str(value)
        parts.append(f"<parameter={key}>{value_str}</parameter>")
    parts.append("</function>")
    return "\n".join(parts)


class OHAgent(BaseAgent):
    """
    OpenHands-compatible Agent for SWE tasks.
    
    This agent uses OpenHands-style tool calling with XML format:
    <function=name><parameter=key>value</parameter></function>
    
    Attributes:
        use_fn_calling: Whether to use OpenAI function calling format.
        format_model_response: Whether to reformat model responses.
        system_prompt: System prompt for the agent.
        tools: List of available tool schemas.
        token_accumulation_enabled: Whether token accumulation is enabled (requires tokenizer and chat_parser).
    """

    def __init__(
        self,
        use_fn_calling: bool = False,
        fn_calling_example: bool = False,
        format_model_response: bool = False,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        tools: list[dict] | None = None,
        use_debug_prompt: bool = True, #TODO: remove this after debugging
        save_completion_text: bool = False,
        save_test_outputs: bool = True,
        enable_token_accumulation: bool = True,
        tokenizer: "PreTrainedTokenizerBase | None" = None,
        chat_parser: "ChatTemplateParser | None" = None,
    ):
        """
        Initialize the OpenHands Agent.
        
        Args:
            use_fn_calling: Whether to use OpenAI function calling format.
            fn_calling_example: Inject tool example when not using fn calling.
            format_model_response: Whether to reformat model responses.
            system_prompt: Custom system prompt. Uses default if None.
            user_prompt_template: Custom user prompt template. Uses default if None.
            tools: Custom tool schemas. Uses default OpenHands tools if None.
            use_debug_prompt: Whether to use simplified debug prompts by default.
            save_completion_text: Whether to save the completion text.
            save_test_outputs: Whether to save reward test outputs for debugging.
            enable_token_accumulation: Whether to enable token accumulation.
            tokenizer: Tokenizer for token accumulation (optional, enables token accumulation if provided).
            chat_parser: Chat template parser for token accumulation (optional, enables token accumulation if provided).
        """
        self.use_fn_calling = use_fn_calling
        self.fn_calling_example = fn_calling_example
        self.format_model_response = format_model_response
        self.use_debug_prompt = use_debug_prompt
        self.save_completion_text = save_completion_text
        self.save_test_outputs = save_test_outputs
        # Token accumulation support
        self.tokenizer = tokenizer
        self.chat_parser = chat_parser
        self.token_accumulation_enabled = enable_token_accumulation and tokenizer is not None and chat_parser is not None
        
        # Use custom or default prompts
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = SYSTEM_PROMPT_DEBUG if use_debug_prompt else OPENHANDS_SYSTEM_PROMPT

        if user_prompt_template is not None:
            self.user_prompt_template = user_prompt_template
        else:
            self.user_prompt_template = USER_PROMPT_DEBUG if use_debug_prompt else OPENHANDS_USER_PROMPT
        
        # Use custom or default tools
        self.tools = tools or get_all_tool_schemas()
        
        # If using function calling, append tool schemas to system prompt
        # TODO: this should be considered in future, now set to false by default
        if self.use_fn_calling:
            self._setup_fn_calling_prompt()
        else:
            if not use_debug_prompt:
                self.system_prompt = append_tools_to_system_prompt(
                    self.system_prompt, self.tools
                )
        
        self._trajectory = Trajectory()
        self.reset()

    def _setup_fn_calling_prompt(self) -> None:
        """Setup system prompt for function calling mode."""
        # Add tool schemas to system prompt for function calling
        tools_json = json.dumps(self.tools, indent=2)
        self.system_prompt = f"""{self.system_prompt}

Available tools (in OpenAI function calling format):
```json
{tools_json}
```
"""

    def process_model_response(self, response: str) -> tuple[str, dict]:
        """
        Process model response to extract thought and action.
        
        Args:
            response: Raw model response.
            
        Returns:
            Tuple of (action_string, info_dict)
        """
        if self.use_fn_calling:
            # Parse OpenAI function calling format
            thought, action = self._parse_oai_response(response)
        else:
            # Parse XML format
            thought, action = self._parse_xml_response(response)
        
        action_str = format_action_xml(action["name"], action["arguments"]) if action["name"] else ""
        
        if self.format_model_response and action_str:
            response = f"{thought}\n\n{action_str}"
            # TODO: this is not used ?
        
        return action_str, {"thought": thought, "action": action}

    def _parse_xml_response(self, response_text: str) -> tuple[str, dict]:
        """
        Parse XML format response.
        
        Returns:
            Tuple of (thought, action_dict)
        """
        # Find function call
        fn_match = re.search(r"<function=\w+>", response_text)
        
        if fn_match:
            # Everything before the function call is the thought
            thought = response_text[:fn_match.start()].strip()
            
            # Parse the function call
            function_name, arguments = parse_xml_function_call(response_text)
            action = {"name": function_name, "arguments": arguments}
        else:
            # No function call found
            thought = response_text.strip()
            action = {"name": "", "arguments": {}}
        
        return thought, action

    def _parse_oai_response(self, response) -> tuple[str, dict]:
        """
        Parse OpenAI function calling format response.
        
        Note: This expects the raw response object from OpenAI API.
        """
        try:
            thought = response.choices[0].message.content or ""
            
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                action = {"name": function_name, "arguments": arguments}
            else:
                action = {"name": "", "arguments": {}}
        except (AttributeError, IndexError, json.JSONDecodeError):
            thought = str(response) if response else ""
            action = {"name": "", "arguments": {}}
        
        return thought, action

    def update_from_env(self, observation, reward, done, info):
        """
        Update agent state after environment step.
        
        Args:
            observation: Observation from environment.
            reward: Reward received.
            done: Whether episode is done.
            info: Additional info from environment.
        """
        observation = str(observation)
        first_observation = False

        # Format first observation with user prompt template
        if not self._trajectory.steps:
            first_observation = True
            try:
                if self.use_debug_prompt:
                    observation = self.user_prompt_template
                else:
                    observation = self.user_prompt_template.format(problem_statement=observation)
                
            except Exception as exc:
                # Match r2egym's format usage, but guard against stray braces in the prompt.
                safe_observation = observation.replace("{", "{{").replace("}", "}}")
                observation = self.user_prompt_template.format(problem_statement=safe_observation)
                logger.warning("Problem statement formatting failed; applied brace escaping: %s", exc)
            if not self.use_fn_calling and self.fn_calling_example:
                observation = append_in_context_example(observation, self.tools)
        
        # Add step information
        max_steps = info.get("max_steps")
        if max_steps:
            remaining = max_steps - self.step - 1
            if remaining > 0:
                # observation += f"\n\nSteps Remaining: {remaining}"
                pass
            else:
                observation += "\n\nYou have reached the maximum number of steps. Please finish your task NOW."
        
        # Check token limit
        cur_tokens = info.get("cur_tokens")
        if cur_tokens is not None and cur_tokens >= TOKEN_WARNING_THRESHOLD:
            observation += "\n\nWarning: You are running out of tokens. Please finish your task NOW."
        
        # Update previous step if exists
        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.next_observation = observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info
        
        # Add observation to messages
        if first_observation:
            self.messages.append({"role": "user", "content": observation})
        else:
            self.messages.append({"role": "tool", "content": observation})
        self.cur_step = Step(observation=observation)

        # Token accumulation: maintain token sequence
        if self.token_accumulation_enabled:
            self._update_tokens_from_env(observation, first_observation)

    def _update_tokens_from_env(self, observation: str, first_observation: bool) -> None:
        """
        Update token sequences after environment step.
        
        For first observation: Initialize init_messages, init_token_ids, curr_token_ids, curr_mask
        For subsequent observations: Append tool message tokens to curr_token_ids
        
        Args:
            observation: The observation content.
            first_observation: Whether this is the first observation in the episode.
        """
        if first_observation:
            # Initialize with system + user messages
            init_messages = list(self.messages)  # Copy current messages (system + user)
            # Encode initial messages with generation prompt to get init_token_ids
            prompt_text = self.chat_parser.parse(
                init_messages, 
                add_generation_prompt=True, 
                is_first_msg=True
            )
            init_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            
            # Initialize curr_token_ids and curr_mask (prompt tokens have mask=0)
            self.curr_token_ids = list(init_token_ids)
            self.curr_mask = [0] * len(init_token_ids)
            
            # Initialize base messages and base token ids (for computing tool message deltas)
            # _base_token_ids_no_gen_prompt is encoded without generation_prompt for proper delta calculation
            self._base_messages = list(self.messages)
            base_text = self.chat_parser.parse(
                self._base_messages,
                add_generation_prompt=False,
                is_first_msg=True
            )
            self._base_token_ids_no_gen_prompt = self.tokenizer.encode(base_text, add_special_tokens=False)
        else:
            # Subsequent observation: append tool message tokens
            # Create the tool message
            tool_message = {"role": "tool", "content": observation}
            
            # Encode _base_messages + tool_message with generation prompt
            # _base_messages includes the full history (system + user + assistant + previous tools)
            messages_with_tool = self._base_messages + [tool_message]
            full_prompt_text = self.chat_parser.parse(
                messages_with_tool,
                add_generation_prompt=True,
                is_first_msg=True
            )
            full_token_ids = self.tokenizer.encode(full_prompt_text, add_special_tokens=False)
            
            # Extract new tokens by removing _base_token_ids_no_gen_prompt prefix
            # This correctly calculates the delta including the tool message and generation prompt
            new_token_ids = full_token_ids[len(self._base_token_ids_no_gen_prompt):]
            
            # Append to curr_token_ids with mask=0 (env feedback)
            self.curr_token_ids.extend(new_token_ids)
            # Note: curr_mask is not used actually, assemble_steps() in agent_execution_engine.py will caculate the mask based on the prompt_ids and completion_ids.
            self.curr_mask.extend([0] * len(new_token_ids))
            
    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Update agent state after model response.
        
        Args:
            response: Model response string.
            **kwargs: Additional arguments.
                - completion_ids: Token IDs of the model completion (for token accumulation).
            
        Returns:
            Action to take in environment.
        """
        self._trajectory.steps.append(self.cur_step)
        
        # Parse response
        if self.use_fn_calling:
            thought, action_dict = self._parse_oai_response(response)
        else:
            thought, action_dict = self._parse_xml_response(response)
        
        action_str = format_action_xml(
            action_dict["name"], 
            action_dict["arguments"]
        ) if action_dict["name"] else ""
        
        # Update current step
        cur_step = self._trajectory.steps[-1]
        cur_step.thought = thought
        cur_step.action = action_str
        cur_step.model_response = response
        
        # Add to messages
        # just for debugging/display purposes
        if self.format_model_response and action_str:
            self.messages.append({"role": "assistant", "content": f"{thought}\n\n{action_str}"})
        else:
            self.messages.append({"role": "assistant", "content": response})
        
        # Token accumulation: append completion_ids with mask=1
        completion_ids = kwargs.get("completion_ids")
        if self.token_accumulation_enabled and completion_ids is not None:
            self._update_tokens_from_model(completion_ids)
        
        self.step += 1
        
        return Action(action=action_dict)

    def _update_tokens_from_model(self, completion_ids: list[int]) -> None:
        """
        Update token sequences after model response.
        
        Appends completion_ids to curr_token_ids with mask=1.
        Also appends trailing tokens if needed (e.g., newline after eos_token).
        
        Args:
            completion_ids: Token IDs of the model completion.
        """
        # Check if we need to append additional tokens after completion_ids
        # vLLM's completion_ids ends with eos_token (e.g., <|im_end|>)
        # but chat template's eot_token might include more (e.g., <|im_end|>\n)
        suffix_ids = []
        if completion_ids and completion_ids[-1] == self.tokenizer.eos_token_id:
            # Get the eot_token from chat_parser
            eot_token = self.chat_parser.eot_token
            eot_ids = self.tokenizer.encode(eot_token, add_special_tokens=False)
            
            # If eot_token has more tokens after eos_token, append them
            # e.g., for Qwen series, eot_token = "<|im_end|>\n" -> eot_ids = [151645, 198]
            # completion_ids ends with 151645, so we need to append 198
            # but the suffix should not be caculated in loss function
            if len(eot_ids) > 1 and eot_ids[0] == self.tokenizer.eos_token_id:
                suffix_ids = eot_ids[1:]  # e.g., [198] for newline
        
        # Append completion tokens with mask=1
        self.curr_token_ids.extend(completion_ids)
        # Note: curr_mask is not used actually, assemble_steps() in agent_execution_engine.py will caculate the mask based on the prompt_ids and completion_ids.
        self.curr_mask.extend([1] * len(completion_ids))
        if suffix_ids:
            self.curr_token_ids.extend(suffix_ids)
            self.curr_mask.extend([0] * len(suffix_ids))

    def get_current_state(self) -> Step | None:
        """Get the current step state."""
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1]

    def reset(self):
        """Reset agent state for new episode."""
        self._trajectory = Trajectory()
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        self.step = 0
        self.cur_step = None
        
        # Token accumulation state
        # These will be initialized in update_from_env when first observation is received

        self.curr_token_ids: list[int] | None = None  # Current accumulated token sequence
        self.curr_mask: list[int] | None = None  # Current mask (1 for model response, 0 for env/prompt)
        self._base_messages: list[dict] | None = None  # Full message history for computing tool deltas
        self._base_token_ids_no_gen_prompt: list[int] | None = None  # Base token IDs without generation prompt

    @property
    def trajectory(self) -> Trajectory:
        """Get the current trajectory."""
        return self._trajectory

    @property
    def chat_completions(self) -> list[dict]:
        """Get the current chat messages."""
        return self.messages

    def get_tool_schemas(self) -> list[dict]:
        """Get the tool schemas for this agent."""
        return self.tools.copy()
    
    # Token accumulation methods
    def get_prompt_token_ids(self) -> list[int]:
        """
        Get the current prompt token IDs for model inference.
        
        Returns:
            The current accumulated token sequence (curr_token_ids).
        
        Raises:
            ValueError: If token accumulation is not enabled or not initialized.
        """
        if not self.token_accumulation_enabled:
            raise ValueError("Token accumulation is not enabled. Provide tokenizer and chat_parser to enable.")
        if self.curr_token_ids is None:
            raise ValueError("Token sequence not initialized. Call update_from_env first.")
        return list(self.curr_token_ids)

