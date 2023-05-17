import onnx
import onnx.checker
import onnx.utils
from onnx.tools import update_model_dims
 
 
model = onnx.load('../onnx_model/shufflenet-v2-10.onnx')
# 此处可以理解为获得了一个维度 “引用”，通过该 “引用“可以修改其对应的维度                                                                                          
dim_proto0 = model.graph.input[0].type.tensor_type.shape.dim[2]
dim_proto3 = model.graph.input[0].type.tensor_type.shape.dim[3]
# 将该维度赋值为字符串，其维度不再为和dummy_input绑定的值
dim_proto0.dim_param = '576'
dim_proto3.dim_param = '1024'
onnx.checker.check_model(model)
print("success!")
onnx.save(model, '../onnx_model/shufflenet-576-1024.onnx')