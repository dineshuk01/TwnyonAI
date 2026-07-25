import json
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, ToolMessage
from agent.state import AgentState
from agent.tools import web_search, run_python, read_memory, write_memory, list_memory, save_to_file, read_file
from memory import mongo

# LLM and tools
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
tools_list = [web_search, run_python, read_memory, write_memory, list_memory, save_to_file, read_file]
llm_with_tools = llm.bind_tools(tools_list)

SYSTEM_PROMPT = """You are Twnyon AI assistant, an autonomous AI agent with persistent memory stored in MongoDB.
You solve tasks step by step using tools.

Available tools:
- web_search(query) - search the web for current information
- run_python(code) - run sandboxed Python for calculations only
- read_memory(key, user_email) - retrieve a stored memory value
- write_memory(key, value, user_email) - save a value to persistent memory
- list_memory(user_email) - list all stored memory keys and values
- save_to_file(filename, content) - save content to a local file
- read_file(filename) - read content from a local file

Strategy:
1. For casual greetings, small talk, or general questions, respond directly and conversationally without calling any tools.
2. Call list_memory or read_memory only when the user's prompt relates to memory, previous context, or stored preferences.
3. Use web_search when you need current or external information.
4. Use write_memory to store or update user preferences, interaction timestamps, facts, or findings for future sessions.
5. Use run_python only for pure calculations or data processing. Never use it for file/OS/shell tasks.
6. Give a friendly, natural, and helpful response.

Use save_to_file and read_file for file operations. Call tools only when necessary."""

async def orchestrator_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    if not messages or messages[0].type != "system":
        from langchain_core.messages import SystemMessage
        user_email = state.get("user_email", "unknown")
        dynamic_prompt = SYSTEM_PROMPT + f"\n\nIMPORTANT: The current user's email is '{user_email}'. You MUST use this email when calling memory tools."
        messages = [SystemMessage(content=dynamic_prompt)] + messages
        
    # Scan for hanging tool calls in history and inject dummy ToolMessages if needed
    scrubbed_messages = []
    for i, msg in enumerate(messages):
        scrubbed_messages.append(msg)
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                found = False
                for j in range(i+1, len(messages)):
                    if isinstance(messages[j], ToolMessage) and messages[j].tool_call_id == tc["id"]:
                        found = True
                        break
                    if isinstance(messages[j], AIMessage):
                        break
                if not found:
                    scrubbed_messages.append(ToolMessage(content="Error: Tool execution crashed or was interrupted.", tool_call_id=tc["id"], name=tc["name"]))
    
    messages = scrubbed_messages

    iteration = state.get("iteration", 0) + 1
    
    if iteration >= 8:
        return {"final_answer": "Max steps reached", "iteration": iteration}
        
    response = await llm_with_tools.ainvoke(messages)
    
    steps = []
    if not response.tool_calls:
        steps.append({"type": "thought", "content": response.content})
        return {"messages": [response], "iteration": iteration, "final_answer": response.content, "steps": steps}
        
    for tc in response.tool_calls:
        steps.append({"type": "action", "tool": tc["name"], "args": tc["args"]})
        
    return {"messages": [response], "iteration": iteration, "steps": steps}

async def tool_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    tool_map = {t.name: t for t in tools_list}
    
    messages = []
    steps = []
    
    for tc in tool_calls:
        tool_name = tc["name"]
        if tool_name not in ["read_memory", "write_memory", "list_memory"]:
            tool_instance = tool_map.get(tool_name)
            if tool_instance:
                try:
                    args = dict(tc["args"])
                    if tool_name in ["save_to_file", "read_file"]:
                        args["user_email"] = state.get("user_email", "")
                    
                    result = await tool_instance.ainvoke(args)
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"], name=tool_name))
                    steps.append({"type": "observation", "tool": tool_name, "result": str(result)})
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    messages.append(ToolMessage(content=error_msg, tool_call_id=tc["id"], name=tool_name))
                    steps.append({"type": "observation", "tool": tool_name, "result": error_msg})
            else:
                messages.append(ToolMessage(content=f"Error: Tool {tool_name} not found.", tool_call_id=tc["id"], name=tool_name))
        else:
            messages.append(ToolMessage(content="Error: Please call memory tools separately from normal tools.", tool_call_id=tc["id"], name=tool_name))
            
    return {"messages": messages, "steps": steps}

async def memory_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    tool_map = {t.name: t for t in tools_list}
    
    messages = []
    steps = []
    
    for tc in tool_calls:
        tool_name = tc["name"]
        if tool_name in ["read_memory", "write_memory", "list_memory"]:
            tool_instance = tool_map.get(tool_name)
            if tool_instance:
                try:
                    user_email = state.get("user_email", "anonymous")
                    args = dict(tc["args"])
                    args["user_email"] = user_email
                    if "session_id" in args:
                        del args["session_id"]
                    
                    result = await tool_instance.ainvoke(args)
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"], name=tool_name))
                    steps.append({"type": "observation", "tool": tool_name, "result": str(result)})
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    messages.append(ToolMessage(content=error_msg, tool_call_id=tc["id"], name=tool_name))
                    steps.append({"type": "observation", "tool": tool_name, "result": error_msg})
            else:
                messages.append(ToolMessage(content=f"Error: Tool {tool_name} not found.", tool_call_id=tc["id"], name=tool_name))
        else:
            messages.append(ToolMessage(content="Error: Please call normal tools separately from memory tools.", tool_call_id=tc["id"], name=tool_name))
            
    return {"messages": messages, "steps": steps}

evaluator_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

async def end_node(state: AgentState) -> AgentState:
    final_answer = state.get("final_answer", "")
    
    eval_prompt = f"""You are a task evaluator. Given a user goal and agent final answer, respond ONLY with this JSON:
{{
  "completed": true or false,
  "score": 1 to 5,
  "suggestion": "one sentence improvement tip"
}}
No other text. No markdown. Raw JSON only.

Final Answer: {final_answer}"""

    from langchain_core.messages import HumanMessage
    try:
        eval_resp = await evaluator_llm.ainvoke([HumanMessage(content=eval_prompt)])
        evaluation = json.loads(eval_resp.content.strip("` \n").replace("json", "", 1))
    except Exception as e:
        evaluation = {"completed": False, "score": 1, "suggestion": f"Evaluation failed: {e}"}
        
    await mongo.save_session(state["session_id"], [m.dict() for m in state["messages"]])
    
    return {"evaluation": evaluation}

def should_continue(state: AgentState) -> str:
    if state.get("final_answer"):
        return "end"
        
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        memory_tools = {"read_memory", "write_memory", "list_memory"}
        has_memory = any(tc["name"] in memory_tools for tc in last_message.tool_calls)
        has_normal = any(tc["name"] not in memory_tools for tc in last_message.tool_calls)
        
        if has_memory and not has_normal:
            return "memory"
        elif has_normal and not has_memory:
            return "tools"
        else:
            return "memory" if has_memory else "tools"
            
    return "orchestrator"
