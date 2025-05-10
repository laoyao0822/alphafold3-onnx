import openvino as ov
import torch

# from TestVino import ov_model

# model = resnet50(weights='DEFAULT')
#
# # prepare input_data
# input_data = torch.rand(1, 3, 224, 224)
#
# ov_model = ov.convert_model(model, example_input=input_data)
# ov_model=ov.convert_model('/root/pycharm/diffusion_head_onnx_base2/diffusion_head.onnx')
onnx_path='/root/pycharm/diffusion_head_onnx_2/diffusion_head.onnx'
vino_path='/root/pycharm/diffusion_head_openvino_2/model.xml'
# onnx_path='/root/pycharm/diffusion_head_onnx_no_aug/diffusion_head.onnx'
# onnx_path='/root/pycharm/evo_onnx/evoformer.onnx'
# vino_path='/root/pycharm/evo_vino/model.xml'
# onnx_path='/root/pycharm/evo_onnx/evoformer_vim.onnx'
# onnx_path='/root/pycharm/sdpa_openvino_vim.onnx'
# vino_path='/root/pycharm/sdpa_openvino.xml'

# internal_path='/root/pycharm/sdpa_inernal.onnx'
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
    # 属性映射（若名称不一致）
    # attr_map={
    #     "scale": "scale",      # 若 ONNX 属性名称为 "scale"，直接映射
    #     "causal": "causal"     # 若 ONNX 属性名称为 "causal"，直接映射
    # },
    # 输入映射（若输入顺序不一致）
    # inputs=["query", "key", "value", "mask"],  # 若输入顺序一致可省略
    # 输出映射（若输出顺序不一致）
    # outputs=["output"]
)

from openvino.frontend import ConversionExtension, NodeContext
def convert_sdpa(node: NodeContext):
    # 提取输入
    query = node.get_input(0)
    key = node.get_input(1)
    value = node.get_input(2)
    mask = node.get_input(3)
    # q_size=node.get_input_size()
    # print(q_size)
    # q_shape=query.shape[-1]
    # scale=q_shape** -0.5
    # 提取属性（假设 ONNX 算子属性名称为 alpha 和 is_causal）
    # scale = node.get_attribute("alpha", 1.0)
    # causal = node.get_attribute("is_causal", False)

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
ov_model=ov.convert_model(onnx_path)
# ov_model=core.read_model(onnx_path)
# ov_model=ov.convert_model(internal_path,verbose=True )
###### Option 1: Save to OpenVINO IR:
print("start to save")
# save model to OpenVINO IR for later use
ov.save_model(ov_model, vino_path,compress_to_fp16=False)

###### Option 2: Compile and infer with OpenVINO:

# # compile model
# compiled_model = ov.compile_model(ov_model)
#
# # run inference
# result = compiled_model(input_data)