import onnx
from onnxsim import simplify

input_path = "mobilenetv2-7.onnx"
output_path = "mobilenetv2_7_sim.onnx"
onnx_model = onnx.load(input_path)  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_path)
print('finished exporting onnx')