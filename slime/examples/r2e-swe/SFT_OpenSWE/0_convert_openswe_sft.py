import json

ROOT_PATH = "/data/user"
INPUT_PATH = f"{ROOT_PATH}/openswe_traj.jsonl"
OUTPUT_PATH = f"{ROOT_PATH}/openswe_sft.jsonl"


def convert_tools_to_system_content(tools: list) -> str:
    """将 tools 列表序列化为 system message 的 content。"""
    return "You have access to the following tools:\n\n" + json.dumps(tools, ensure_ascii=False)


def convert_record(record: dict) -> dict:
    messages = []

    # 1. system message: 从 messages 里取 role==system 的 content
    for m in record["messages"]:
        if m["role"] == "system":
            messages.append({"role": "system", "content": m["content"], "loss_mask": 0})
            break

    # 2. system message: tools 定义
    messages.append({
        "role": "system",
        "content": convert_tools_to_system_content(record["tools"]),
        "loss_mask": 0,
    })

    # 3. 遍历对话消息（跳过 system，已处理）
    for m in record["messages"]:
        role = m["role"]
        loss_mask = 1 if m.get("calculate_loss") is True else 0

        if role == "system":
            continue

        elif role == "user":
            messages.append({"role": "user", "content": m["content"], "loss_mask": loss_mask})

        elif role == "assistant":
            content = m.get("content", "") or ""
            if "<think>" not in content[:20]:
                content = "\n<think>\n" + content
            if "</think>" not in content[-20:]:
                content = content + "\n</think>"
            tool_calls = m.get("tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    content += "\n<tool_call>\n" + json.dumps(tc, ensure_ascii=False) + "\n</tool_call>"
            messages.append({"role": "assistant", "content": content, "loss_mask": loss_mask})

        elif role == "tool":
            wrapped = "<tool_response>\n" + m["content"] + "\n</tool_response>"
            messages.append({"role": "tool", "content": wrapped, "loss_mask": loss_mask})

    return {"input": messages}


def main():
    with open(INPUT_PATH) as f:
        records = [json.loads(l) for l in f]

    print(f"读取记录数: {len(records)}")

    converted = [convert_record(r) for r in records]

    with open(OUTPUT_PATH, "w") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"输出到: {OUTPUT_PATH}")

    # 打印第一条样本结构预览
    sample = converted[0]
    print(f"\n第一条样本消息数: {len(sample['input'])}")
    for i, msg in enumerate(sample["input"][:6]):
        role = msg["role"]
        content_preview = repr(str(msg.get("content", ""))[:800])
        print(f"  [{i}] role={role}, content={content_preview}")


if __name__ == "__main__":
    main()
