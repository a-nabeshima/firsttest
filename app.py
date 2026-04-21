import asyncio
import json
import os

import boto3
import streamlit as st
from dotenv import load_dotenv

from mcp_client import EAMcpClient

load_dotenv()

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
)
EA_MCP_COMMAND_STR = os.getenv(
    "EA_MCP_COMMAND",
    r"C:\Program Files\Sparx Systems\EA\MCP_Server\MCP3.exe",
)
EA_COMMAND = EA_MCP_COMMAND_STR.split()
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-1")

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# EA MCP クライアント（Streamlit セッション単位でキャッシュ）
# ---------------------------------------------------------------------------
@st.cache_resource
def get_ea_client() -> EAMcpClient:
    client = EAMcpClient(EA_COMMAND)
    asyncio.run(client.start())
    return client


# ---------------------------------------------------------------------------
# MCP tools → Bedrock Converse toolConfig 変換
# MCP: {"inputSchema": {...}}
# Bedrock: {"toolSpec": {"inputSchema": {"json": {...}}}}
# ---------------------------------------------------------------------------
def build_tool_config(tools: list[dict]) -> dict:
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "inputSchema": {
                        "json": t.get(
                            "inputSchema", {"type": "object", "properties": {}}
                        )
                    },
                }
            }
            for t in tools
        ]
    }


# ---------------------------------------------------------------------------
# Bedrock Converse API + tool_use ループ
# ---------------------------------------------------------------------------
def run_agent(ea: EAMcpClient, messages: list[dict]) -> str:
    tool_config = build_tool_config(ea.tools)

    while True:
        resp = bedrock.converse(
            modelId=MODEL_ID,
            messages=messages,
            toolConfig=tool_config,
            inferenceConfig={"maxTokens": 4096, "temperature": 0.1},
        )
        out_msg = resp["output"]["message"]
        messages.append(out_msg)
        stop = resp["stopReason"]

        if stop == "end_turn":
            for block in out_msg["content"]:
                if "text" in block:
                    return block["text"]
            return ""

        if stop == "tool_use":
            tool_results = []
            for block in out_msg["content"]:
                if "toolUse" not in block:
                    continue
                tu = block["toolUse"]
                try:
                    result = asyncio.run(ea.call_tool(tu["name"], tu["input"]))
                    # MCP tools/call result は content 配列を持つ
                    content_text = json.dumps(
                        result.get("content", result), ensure_ascii=False
                    )
                except Exception as exc:
                    content_text = f"Error calling {tu['name']}: {exc}"
                tool_results.append(
                    {
                        "toolUseId": tu["toolUseId"],
                        "content": [{"text": content_text}],
                    }
                )
            # tool_use 結果は "user" ロールで返す（Bedrock の仕様）
            messages.append(
                {
                    "role": "user",
                    "content": [{"toolResult": tr} for tr in tool_results],
                }
            )
        else:
            # max_tokens / content_filtered など
            return f"[stopped: {stop}]"


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="EA AI Agent", page_icon="🏗️", layout="wide")
st.title("🏗️ EA AI Agent  (Bedrock + MCP)")

# EA MCP 接続
try:
    ea = get_ea_client()
    st.sidebar.success(f"EA MCP 接続済み — {len(ea.tools)} ツール利用可能")
    with st.sidebar.expander("利用可能なツール"):
        for t in ea.tools:
            desc = t.get("description", "")[:80]
            st.markdown(f"- **{t['name']}**: {desc}")
except Exception as exc:
    st.sidebar.error(f"EA MCP 接続エラー: {exc}")
    st.error(
        "Enterprise Architect が起動していないか、MCP3.exe のパスが正しくありません。"
        "\n\n`.env` の `EA_MCP_COMMAND` を確認してください。"
    )
    st.stop()

# 会話履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 会話履歴の表示
for m in st.session_state.messages:
    role = m["role"]
    text = next((b["text"] for b in m["content"] if "text" in b), None)
    if text and role in ("user", "assistant"):
        with st.chat_message(role):
            st.markdown(text)

# ユーザー入力
if user_input := st.chat_input("Enterprise Architect への指示を入力..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(
        {"role": "user", "content": [{"text": user_input}]}
    )

    with st.chat_message("assistant"):
        with st.spinner("EA を操作中..."):
            reply = run_agent(ea, st.session_state.messages)
        st.markdown(reply)
