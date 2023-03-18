import onnx


def print_node(model= "../yolopu.onnx"):
    node_list =[]
    op_name_list =- []
    model= onnx.load(model)
    onnx.checker.check_model(model)
    # 获得onnx图
    print("Sucessfully load!")
    graph = model.graph

    for node_id,node in enumerate(graph.node):
        node_list.append(node)
        op_name_list.append(node.name)
        print("######%s######" % node_id)
        print(node)
print_node('../yolopl.onnx')