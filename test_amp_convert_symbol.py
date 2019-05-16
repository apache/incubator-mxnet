import mxnet as mx
from gluoncv.model_zoo import get_model

# Full Model
FP16_FUNCS = ["Convolution",
              "Deconvolution",
              "FullyConnected",
              "RNN"]
FP32_FUNCS = [
'arccos',
'arcsin',
'cosh',
'erfinv',
'sinh',
'tan',
# Exponents
'exp',
'expm1',
'log',
'log10',
'log2',
'log1p',
# Powers
'broadcast_pow',
'broadcast_power',
'square',
'reciprocal',
'rsqrt',
'rcbrt',
'__pow__',
'pow',
'linalg_sumlogdiag',
'hypot',
'broadcast_hypot',
# Reductions
'sum',
'nansum',
'prod',
'nanprod',
'mean',
'norm',
'softmin',
# Misc
'gamma',
'gammaln',
'linalg_syrk',
'linalg_potrf',
'linalg_gemm2',
'linalg_gelqf',
'linalg_trmm',
'linalg_trsm',
'quantize',
'quantize_v2',
# Neural network
'SoftmaxOutput',
'softmax',
'log_softmax',
'InstanceNorm',
'LayerNorm',
'L2Normalization',
'LRN',
'SoftmaxActivation',
'LinearRegressionOutput',
'LogisticRegressionOutput',
'MAERegressionOutput',
'SVMOutput',
'softmax_cross_entropy',
'smooth_l1',
'MakeLoss',
'make_loss',
'Custom',
'CTCLoss',
'ctc_loss',
'DeformableConvolution'
'DeformablePSROIPooling',
'SyncBatchNorm']

WIDEST_TYPE_CASTS = [
'__add__',
'__sub__',
'__rsub__',
'__mul__',
'__div__',
'__rdiv__',
'__mod__',
'__rmod__',
'__ne__',
'__eq__',
'__gt__',
'__ge__',
'__lt__',
'__le__',
'concat',
'Concat',
'Correlation',
'ElementWiseSum',
'add_n',
'batch_dot',
'broadcast_add',
'broadcast_plus',
'broadcast_div',
'broadcast_equal',
'broadcast_greater',
'broadcast_greater_equal',
'broadcast_lesser',
'broadcast_lesser_equal',
'broadcast_logical_and',
'broadcast_logical_or',
'broadcast_logical_xor',
'broadcast_maximum',
'broadcast_minimum',
'broadcast_minus',
'broadcast_mod',
'broadcast_mul',
'broadcast_not_equal',
'broadcast_sub',
'dot',
'elemwise_add',
'elemwise_div',
'elemwise_mul',
'elemwise_sub',
'stack',
'maximum',
'minimum',
'MultiBoxDetection',
'MultiBoxTarget',
'MultiProposal',
'PSROIPooling',
'Proposal',
'ROIAlign',
'boolean_mask',
'box_iou',
'count_sketch',
'dgl_csr_neighbor_non_uniform_sample',
'dgl_csr_neighbor_uniform_sample',
'dgl_graph_compact',
'dgl_subgraph',
'edge_id',
'where',
'_rnn_concat_param',
]

'''
sym, arg_params, aux_params = mx.model.load_checkpoint("resnet18", 0)
result_sym, arg_params, aux_params = mx.contrib.amp.convert_model(sym, arg_params, aux_params, target_dtype="float16",
                                                                  target_dtype_ops=FP16_FUNCS, fp32_ops=FP32_FUNCS, widest_dtype_ops=WIDEST_TYPE_CASTS)
mod = mx.mod.Module(result_sym, data_names=['data'], context=mx.gpu(0))
mod.bind(data_shapes=[['data', (1, 3, 224, 224)]])
for key in mod._arg_params:
    print(key)
    print(mod._arg_params[key].dtype)
mod.set_params(arg_params, aux_params)
mod.forward(mx.io.DataBatch(data=[mx.nd.ones((1, 3, 224, 224))],
                            label=[mx.nd.ones((1,))]))
result = mod.get_outputs()[0].asnumpy()
sym, arg_params, aux_params = mx.model.load_checkpoint("imagenet1k-resnet-152", 0)
result_sym, arg_params, aux_params = mx.contrib.amp.convert_model(sym, arg_params, aux_params, target_dtype="float16",
                                                                  target_dtype_ops=FP16_FUNCS, fp32_ops=FP32_FUNCS, widest_dtype_ops=WIDEST_TYPE_CASTS)
'''
path='http://data.mxnet.io/models/imagenet/'
[mx.test_utils.download(path+'resnet/18-layers/resnet-18-0000.params'),
mx.test_utils.download(path+'resnet/18-layers/resnet-18-symbol.json'),
mx.test_utils.download(path+'synset.txt')]
'''
[mx.test_utils.download(path+'resnet/50-layers/resnet-50-0000.params'),
        mx.test_utils.download(path+'resnet/50-layers/resnet-50-symbol.json'),
        mx.test_utils.download(path+'synset.txt')]
'''

sym, arg_params, aux_params = mx.model.load_checkpoint("resnet-18", 0)
result_sym, arg_params, aux_params = mx.contrib.amp.convert_model(sym, arg_params, aux_params, target_dtype="float16",
                                                                          target_dtype_ops=FP16_FUNCS, fp32_ops=FP32_FUNCS, widest_dtype_ops=WIDEST_TYPE_CASTS)
mod = mx.mod.Module(result_sym, data_names=['data'], label_names=['softmax_label'], context=mx.gpu(0))
#mod = mx.mod.Module(sym, data_names=['data'], context=mx.gpu(0))
#mod = mx.mod.Module(result_sym, data_names=['data'], context=mx.gpu(0))
#mod.bind(data_shapes=[['data', (1, 3, 224, 224)]], label_shapes=[['softmax_label', (1,)]])
mod.bind(data_shapes=[['data', (1, 3, 224, 224)]])
mod.set_params(arg_params, aux_params)
for key in arg_params.keys():
    print(arg_params[key].dtype)
for key in aux_params.keys():
    print(aux_params[key].dtype)
mod.forward(mx.io.DataBatch(data=[mx.nd.ones((1, 3, 224, 224))],
                            label=[mx.nd.ones((1,))]))
result = mod.get_outputs()[0].asnumpy()
mod._symbol.save("after.json")
