import re
from graph import edges, nodes

sNodes = []
newLine = "newLine"
queueNode = [nodes[0], newLine]
print(queueNode)


yVal = 0
xVal = 0

while(len(queueNode) > 1):
    if(queueNode[0] == newLine):
        yVal = yVal - 1
        xVal = 0
        queueNode.append(queueNode.pop(0))
    cNode = queueNode.pop(0)
    sNodes.append(re.sub(r"(?<=at \()[^)]*", f"{xVal}, {yVal}", cNode))
    match = re.search(r"\(([^)]+)\)", cNode)
    cNode = match.group(1) if match else "error"

    children = [e for e in edges if "(" + cNode + ") --" in e]
    copy = []
    for child in children:
        match = re.search(r"\) -- \(([^)]*)\)", child)
        child = match.group(1) if match else "error"
        if child != "error":
            copy.append(child)

    children = list(set(copy))

    for child in children:
        print("child")
        print(child)
        matchedNodes = [node for node in nodes if child in node]
        if len(matchedNodes) > 0:
            matchedNode = matchedNodes[0]
            nodes.remove(matchedNode)
            queueNode.append(matchedNode)
    xVal = xVal + 6

print("Sorted Nodes")
print(sNodes)