import onnx

# 导入resnet50.onnx模型
resnet50_onnx = onnx.load("./resnet18_224_sim.onnx")
# 获得onnx图
graph = resnet50_onnx.graph
# 获得onnx节点
node = graph.node

### 准备工作已就绪，开干
# 增、删、改、查一起操作
# 比如咱们要对 `算子类型为Add&输出为225的节点` 进行操作
# for i in range(len(node)):
#     if node[i].op_type == 'Add':
#         node_rise = node[i]
#         if node_rise.output[0] == '225':
#             print(i)  # 169 => 查到这个算子的ID为169

old_node = node[-1]  # 定位到刚才检索到的算子

# 新增一个 `Constant` 算子
# new_node = onnx.helper.make_node(
#     "Constant",
#     inputs=[],
#     outputs=['225'],
#     value=onnx.helper.make_tensor('value', onnx.TensorProto.FLOAT, [4], [1, 1, 1.2, 1.2])
# ) 

# 删除旧节点
node.remove(old_node)  

# 插入新节点
# node.insert(169, new_node)  

# 是不是还少一个修改节点，比方看下面
# node[169].type = 'Conv'   # 将刚才的算子类型改为2D卷积
# 改名称啥的类似

### 保存新模型
# 校验
# onnx.checker.check_model(resnet50_onnx)
# 保存
onnx.save(resnet50_onnx,'new.onnx')
for node_id,node in enumerate(graph.node):
    print("######%s######" % node_id)
    print(node)