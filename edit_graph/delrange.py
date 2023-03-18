import onnx  
# from print_node import print_node


model_ori = '../resnet18_224_sim.onnx'
model_new = '../resnet18_224_sim_new3.onnx'

# print_node(model_ori)

model = onnx.load(model_ori) 
graph = model.graph
# node_id = 46

for node_id,node in enumerate(graph.node):
    if node_id>=46:
        old_squeeze_node = graph.node[node_id]
        graph.node.remove(old_squeeze_node)
# to_leave = graph.node[node_id-1]
# to_edit = graph.node[-1]
# print(to_leave.output[0])
# print(to_edit.input[0])
# to_edit.input[0] = to_leave.output[0]

print("del",node_id)
# onnx.checker.check_model(model)
onnx.save(model,model_new)