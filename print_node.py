import onnx

def print_node(model= "./resnet18_224_sim_new.onnx"):
    model= onnx.load(model)
    # 获得onnx图
    graph = model.graph

    for node_id,node in enumerate(graph.node):
        print("######%s######" % node_id)
        print(node)
print_node()