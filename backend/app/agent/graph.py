"""
Wires the nodes into a linear LangGraph StateGraph:

    profile -> plan -> feature_eng -> train -> explain -> report -> END

It's linear today on purpose (a portfolio MVP should be reliable and easy
to demo). The natural "stretch" extension is to make `plan_node` route
conditionally -- e.g. a time-series branch that goes to a Prophet/ARIMA
node instead of train_node -- using LangGraph's conditional edges. The
graph shape is exactly what makes that extension a small diff instead of
a rewrite.
"""
from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    explain_node,
    feature_eng_node,
    plan_node,
    profile_node,
    report_node,
    train_node,
)
from app.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("profile", profile_node)
    graph.add_node("plan", plan_node)
    graph.add_node("feature_eng", feature_eng_node)
    graph.add_node("train", train_node)
    graph.add_node("explain", explain_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("profile")
    graph.add_edge("profile", "plan")
    graph.add_edge("plan", "feature_eng")
    graph.add_edge("feature_eng", "train")
    graph.add_edge("train", "explain")
    graph.add_edge("explain", "report")
    graph.add_edge("report", END)

    return graph.compile()


# Compiled once at import time and reused across requests.
compiled_graph = build_graph()
