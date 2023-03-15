import onnx

input_path = '../yolopu.onnx'
output_path = '../yolopu1.onnx'
input_names = ['images']
output_names = ['det_out']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)