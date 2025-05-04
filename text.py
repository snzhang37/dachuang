import json
import pandas as pd

def json_to_excel(json, file_name):
    # 读取 JSON 文件
    with open(json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取节点数据
    nodes = data.get("nodes", [])
    # 如果节点的坐标存储在 pos 列表中，可以将其拆分为 x 和 y 两列
    nodes_df = pd.DataFrame(nodes)
    if "pos" in nodes_df.columns:
        nodes_df[["x", "y"]] = pd.DataFrame(nodes_df["pos"].tolist(), index=nodes_df.index)
        nodes_df = nodes_df.drop(columns=["pos"])

    # 保存节点信息到 Excel 或 CSV 文件
    nodes_df.to_excel("new_example/"+file_name+".node.xlsx", index=False)

    # 提取链路数据（假设 JSON 中有 "links" 字段）
    links = data.get("links", [])
    links_df = pd.DataFrame(links)

    # 保存链路信息到 Excel 或 CSV 文件
    links_df.to_excel("new_example/"+file_name+".oms.xlsx", index=False)

