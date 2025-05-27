import os

import openvino as ov

from openvino.frontend import OpExtension
# import onnx
# from onnx import helper, shape_inference
#
# # 加载原始 ONNX 模型
# model = onnx.load(onnx_path)
#
# # 遍历所有节点，找到自定义算子节点
# nodes_to_replace = []
# for node in model.graph.node:
#     if node.op_type == "scaled_dot_product_attention" and node.domain == "openvino":
#         nodes_to_replace.append(node)
#
# # 批量替换节点
# for old_node in nodes_to_replace:
#     # 获取原节点在列表中的位置
#     index = list(model.graph.node).index(old_node)
#
#     # 创建新的 Identity 节点（保留所有输入和属性）
#     new_node = helper.make_node(
#         "Identity",
#         inputs=old_node.input,  # 保留原始输入
#         outputs=old_node.output,  # 保持输出名称不变
#         name=old_node.name + "_placeholder"
#     )
#
#     # 替换节点
#     model.graph.node.remove(old_node)
#     model.graph.node.insert(index, new_node)
#
# # 更新模型结构和形状推断
# model = shape_inference.infer_shapes(model)
# onnx.save(model, internal_path)




sdpa_extension = OpExtension(
    ov_type_name="ScaledDotProductAttention",  # OpenVINO 内置算子名称
    fw_type_name="openvino.scaled_dot_product_attention",  # ONNX 中自定义算子名称
    attr_names_map={
        "scale": "scale"
    },
    # 设置默认属性值
    attr_values_map={
        "causal": False  # 默认非因果注意力
    }
)

from openvino.frontend import ConversionExtension, NodeContext
def convert_sdpa(node: NodeContext):
    # 提取输入
    query = node.get_input(0)
    key = node.get_input(1)
    value = node.get_input(2)
    mask = node.get_input(3)
    # 创建 OpenVINO 的 SDPA 算子
    sdpa = ov.opset15.scaled_dot_product_attention(query, key, value, mask,scale=None)
    return sdpa.outputs()

from openvino import Core
from openvino.frontend.onnx import ConversionExtension
from openvino.frontend.onnx import OpExtension
# core = Core()
# core.add_extension(ConversionExtension("scaled_dot_product_attention", "openvino", convert_sdpa))
#

# core.add_extension(sdpa_extension)

# 注册转换扩展
# core.add_extension(ConversionExtension("ai.onnx.Identity", convert_sdpa))
# 添加扩展到 OpenVINO 运行时

OPENVINO_PATH = '/root/asc25'
DIFFUSION_OPENVINO_PATH=OPENVINO_PATH+'/diffusion_head_openvino'
EVO_VINO_PATH=OPENVINO_PATH+'/evo_vino'
CONFIDENCE_VINO_PATH=OPENVINO_PATH+'/confidence_vino'
os.makedirs(CONFIDENCE_VINO_PATH, exist_ok=True)
os.makedirs(EVO_VINO_PATH, exist_ok=True)
os.makedirs(DIFFUSION_OPENVINO_PATH, exist_ok=True)


ONNX_PATH='/root/asc25'
DIFFUSION_ONNX_PATH = ONNX_PATH + '/diffusion_head_onnx/diffusion_head.onnx'
EVO_ONNX_PATH = ONNX_PATH + '/evo_onnx/evoformer.onnx'
CONFIDENCE_ONNX_PATH = ONNX_PATH + '/confidence_onnx/confidence_head.onnx'

print("start to convert evoformer vino model")
evo_vino=ov.convert_model(EVO_ONNX_PATH)
print("start to save evo vino model")
ov.save_model(evo_vino, EVO_VINO_PATH+"/model.xml",compress_to_fp16=True)
print("start to convert diffusion vino model")
diffusion_vino=ov.convert_model(DIFFUSION_ONNX_PATH)
print("start to save diffusion vino model")
ov.save_model(diffusion_vino, DIFFUSION_OPENVINO_PATH+"/model.xml",compress_to_fp16=True)
print("start to convert confidence vino model")
confidence_vino=ov.convert_model(CONFIDENCE_ONNX_PATH)
print("start to save confidence vino model")
ov.save_model(confidence_vino, CONFIDENCE_VINO_PATH+"/model.xml",compress_to_fp16=True)
print("all model have convert to openvino")


