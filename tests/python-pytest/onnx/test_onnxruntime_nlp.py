# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
import numpy as np
import onnxruntime

from mxnet.test_utils import assert_almost_equal
from common import with_seed

import json
import os
import pytest
import shutil


@with_seed()
@pytest.mark.parametrize('model_name', ['roberta_24_1024_16', 'roberta_12_768_12'])
@pytest.mark.parametrize('model_specific_logics', [None, 'gluonnlp_bert_uninterleaved'])
def test_roberta_inference_onnxruntime(tmp_path, model_name, model_specific_logics):
    tmp_path = str(tmp_path)
    try:
        import gluonnlp as nlp
        ctx = mx.cpu(0)

        dataset= 'openwebtext_ccnews_stories_books_cased'#'book_corpus_wiki_en_uncased'
        model, _ = nlp.model.get_model(
        name=model_name,
        ctx=ctx,
        pretrained=True,
        use_decoder=True,
        dataset_name=dataset)
        
        model.hybridize(static_alloc=False)

        batch = 2
        seq_length = 32
        num_masked_positions = 1
        inputs = mx.nd.random.uniform(0, 30522, shape=(batch, seq_length), dtype='float32', ctx=ctx)
        valid_length = mx.nd.array([seq_length] * batch, dtype='float32', ctx=ctx)
        masked_positions = mx.nd.random.uniform(0, 32, shape=(batch, num_masked_positions),
            dtype='float32', ctx=ctx).astype('int32')

        sequence_outputs, attention_outputs= model(inputs, valid_length, masked_positions)    

        prefix = "%s/roberta" % tmp_path
        model.export(prefix)

        sym_file = "%s-symbol.json" % prefix
        params_file = "%s-0000.params" % prefix
        onnx_file = "%s.onnx" % prefix
        input_shapes = [(batch, seq_length), (batch,), (batch, num_masked_positions)]

        # this is only for when model_specific_logics='gluonnlp_bert_uninterleaved'
        cheat_sheet = {
            'qkv_hidden': 768 if model_name == 'roberta_12_768_12' else 1024,
            'num_heads': 12 if model_name == 'roberta_12_768_12' else 16,
            'head_dim': 64
        }

        converted_model_path = mx.onnx.export_model(sym_file, params_file, input_shapes,
                                                    [np.float32, np.float32, np.int32],
                                                    onnx_file, verbose=True,
                                                    model_specific_logics=model_specific_logics,
                                                    cheat_sheet=cheat_sheet)

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = onnxruntime.InferenceSession(onnx_file, sess_options)

        in_tensors = [inputs, valid_length, masked_positions]
        input_dict = dict((sess.get_inputs()[i].name, in_tensors[i].asnumpy()) for i in range(len(in_tensors)))
        pred = sess.run(None, input_dict)

        assert_almost_equal(sequence_outputs, pred[0])
        assert_almost_equal(attention_outputs, pred[1])

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.integration
@pytest.mark.parametrize('model_name', ['bert_12_768_12', 'bert_24_1024_16'])
@pytest.mark.parametrize('model_specific_logics', [None, 'gluonnlp_bert_uninterleaved'])
def test_bert_inference_onnxruntime(tmp_path, model_name, model_specific_logics):
    tmp_path = str(tmp_path)
    try:
        import gluonnlp as nlp
        dataset = 'book_corpus_wiki_en_uncased'
        ctx = mx.cpu(0)
        model, vocab = nlp.model.get_model(
            name=model_name,
            ctx=ctx,
            dataset_name=dataset,
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            use_classifier=False)

        model.hybridize(static_alloc=True)

        batch = 5
        seq_length = 16
        # create synthetic test data
        inputs = mx.nd.random.uniform(0, 30522, shape=(batch, seq_length), dtype='float32')
        token_types = mx.nd.random.uniform(0, 2, shape=(batch, seq_length), dtype='float32')
        valid_length = mx.nd.array([seq_length] * batch, dtype='float32')

        seq_encoding, cls_encoding = model(inputs, token_types, valid_length)

        prefix = "%s/bert" % tmp_path
        model.export(prefix)
        sym_file = "%s-symbol.json" % prefix
        params_file = "%s-0000.params" % prefix
        onnx_file = "%s.onnx" % prefix

        # this is only for when model_specific_logics='gluonnlp_bert_uninterleaved'
        cheat_sheet = {
            'qkv_hidden': 768 if model_name == 'bert_12_768_12' else 1024,
            'num_heads': 12 if model_name == 'bert_12_768_12' else 16,
            'head_dim': 64
        }

        input_shapes = [(batch, seq_length), (batch, seq_length), (batch,)]
        input_types = [np.float32, np.float32, np.float32]
        converted_model_path = mx.onnx.export_model(sym_file, params_file, input_shapes,
                                                    input_types, onnx_file,
                                                    model_specific_logics=model_specific_logics,
                                                    cheat_sheet=cheat_sheet)


        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)
        onnx_inputs = [inputs, token_types, valid_length]
        input_dict = dict((session.get_inputs()[i].name, onnx_inputs[i].asnumpy()) for i in range(len(onnx_inputs)))
        pred_onx, cls_onx = session.run(None, input_dict)

        assert_almost_equal(seq_encoding, pred_onx, rtol = 0.0001, atol = 0.0025)
        assert_almost_equal(cls_encoding, cls_onx, rtol = 0.0001, atol = 0.0025)

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.parametrize('model_name', ['distilbert_6_768_12'])
@pytest.mark.parametrize('model_specific_logics', [None, 'gluonnlp_bert_uninterleaved'])
def test_distilbert_inference_onnxruntime(tmp_path, model_name, model_specific_logics):
    tmp_path = str(tmp_path)
    try:
        import gluonnlp as nlp
        dataset = 'distilbert_book_corpus_wiki_en_uncased'
        ctx = mx.cpu(0)
        model, _ = nlp.model.get_model(
            name=model_name,
            ctx=ctx,
            pretrained=True,
            dataset_name=dataset)

        model.hybridize(static_alloc=True)

        batch = 2
        seq_length = 32
        num_masked_positions = 1
        inputs = mx.nd.random.uniform(0, 30522, shape=(batch, seq_length), dtype='float32', ctx=ctx)
        valid_length = mx.nd.array([seq_length] * batch, dtype='float32', ctx=ctx)

        sequence_outputs = model(inputs, valid_length)

        prefix = "%s/distilbert" % tmp_path
        model.export(prefix)
        sym_file = "%s-symbol.json" % prefix
        params_file = "%s-0000.params" % prefix
        onnx_file = "%s.onnx" % prefix

        input_shapes = [(batch, seq_length), (batch,)]
        # this is only for when model_specific_logics='gluonnlp_bert_uninterleaved'
        cheat_sheet = {
            'qkv_hidden': 768,
            'num_heads': 12,
            'head_dim': 64
        }
        converted_model_path = mx.onnx.export_model(sym_file, params_file, input_shapes,
                                                    [np.float32, np.float32],
                                                    onnx_file, verbose=True,
                                                    model_specific_logics=model_specific_logics,
                                                    cheat_sheet=cheat_sheet)
        
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = onnxruntime.InferenceSession(onnx_file, sess_options)

        in_tensors = [inputs, valid_length]
        input_dict = dict((sess.get_inputs()[i].name, in_tensors[i].asnumpy()) for i in range(len(in_tensors)))
        pred = sess.run(None, input_dict)

        assert_almost_equal(sequence_outputs, pred[0])

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.parametrize('model_name', [('standard_lstm_lm_200', 200), ('standard_lstm_lm_650', 650),
                                        ('standard_lstm_lm_1500', 1500)])
@pytest.mark.parametrize('seq_length', [64, 128])
def test_standard_rnn_lstm_pretrained_inference_onnxruntime(tmp_path, model_name, seq_length):
    try:
        import gluonnlp as nlp
        ctx = mx.cpu()
        dataset= 'wikitext-2'
        model, _ = nlp.model.get_model(
            name=model_name[0],
            ctx=ctx,
            pretrained=True,
            dataset_name=dataset,
            dropout=0)
        model.hybridize()

        batch = 2
        num_hidden = model_name[1]
        num_layers = 2
        inputs = mx.nd.random.randint(0, 33278, shape=(seq_length, batch),
                                      ctx=ctx).astype('float32')
        begin_state = model.begin_state(func=mx.nd.random.uniform, low=0, high=1,
                                        batch_size=batch, dtype='float32', ctx=ctx)
        out, out_state= model(inputs, begin_state)

        prefix = "%s/standard_rnn_lstm" % tmp_path
        model.export(prefix)
        sym_file = "%s-symbol.json" % prefix
        params_file = "%s-0000.params" % prefix
        onnx_file = "%s.onnx" % prefix

        input_shapes = [(seq_length, batch), np.shape(begin_state[0]), np.shape(begin_state[1])]
        converted_model_path = mx.onnx.export_model(sym_file, params_file, input_shapes,
                                                    [np.float32, np.float32, np.float32],
                                                    onnx_file, verbose=True)
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = onnxruntime.InferenceSession(onnx_file, sess_options)

        in_tensors = [inputs, begin_state[0], begin_state[1]]
        input_dict = dict((sess.get_inputs()[i].name, in_tensors[i].asnumpy()) for i in range(len(in_tensors)))
        pred = sess.run(None, input_dict)

        assert_almost_equal(out, pred[2])
        assert_almost_equal(out_state[0], pred[0])
        assert_almost_equal(out_state[1], pred[1])

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.integration
@pytest.mark.parametrize('model', ['bert_12_768_12'])
def test_dynamic_shape_bert_inference_onnxruntime(tmp_path, model):
    tmp_path = str(tmp_path)
    try:
        import gluonnlp as nlp
        dataset = 'book_corpus_wiki_en_uncased'
        ctx = mx.cpu(0)
        model, vocab = nlp.model.get_model(
            name=model,
            ctx=ctx,
            dataset_name=dataset,
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            num_layers = 3,
            hparam_allow_override = True,
            use_classifier=False)

        model.hybridize(static_alloc=True)

        batch = 5
        seq_length = 16
        # create synthetic test data
        inputs = mx.nd.random.uniform(0, 30522, shape=(batch, seq_length), dtype='float32')
        token_types = mx.nd.random.uniform(0, 2, shape=(batch, seq_length), dtype='float32')
        valid_length = mx.nd.array([seq_length] * batch, dtype='float32')

        seq_encoding, cls_encoding = model(inputs, token_types, valid_length)

        prefix = "%s/bert" % tmp_path
        model.export(prefix)
        sym_file = "%s-symbol.json" % prefix
        params_file = "%s-0000.params" % prefix
        onnx_file = "%s.onnx" % prefix

        dynamic_input_shapes = [(None, seq_length), (None, seq_length), (None,)]
        input_shapes = [(batch, seq_length), (batch, seq_length), (batch,)]
        input_types = [np.float32, np.float32, np.float32]
        converted_model_path = mx.onnx.export_model(sym_file, params_file, input_shapes,
                                                    input_types, onnx_file,
                                                    dynamic=True,
                                                    dynamic_input_shapes=dynamic_input_shapes)

        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)

        # test on a different batch size
        batch = 7
        seq_length = 16
        inputs = mx.nd.random.uniform(0, 30522, shape=(batch, seq_length), dtype='float32')
        token_types = mx.nd.random.uniform(0, 2, shape=(batch, seq_length), dtype='float32')
        valid_length = mx.nd.array([seq_length] * batch, dtype='float32')

        seq_encoding, cls_encoding = model(inputs, token_types, valid_length)

        onnx_inputs = [inputs, token_types, valid_length]
        input_dict = dict((session.get_inputs()[i].name, onnx_inputs[i].asnumpy()) for i in range(len(onnx_inputs)))
        pred_onx, cls_onx = session.run(None, input_dict)

        assert_almost_equal(seq_encoding, pred_onx)
        assert_almost_equal(cls_encoding, cls_onx)

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.parametrize('model_name', [('awd_lstm_lm_600', 600), ('awd_lstm_lm_1150', 1150)])
@pytest.mark.parametrize('seq_length', [16, 128, 256])
def test_awd_rnn_lstm_pretrained_inference_onnxruntime(tmp_path, model_name, seq_length):
    try:
        import gluonnlp as nlp
        ctx = mx.cpu()
        dataset= 'wikitext-2'
        model, _ = nlp.model.get_model(
            name=model_name[0],
            ctx=ctx,
            pretrained=True,
            dataset_name=dataset,
            dropout=0)
        model.hybridize()

        batch = 2
        num_hidden = model_name[1]
        num_layers = 2
        inputs = mx.nd.random.randint(0, 33278, shape=(seq_length, batch),
                                      ctx=ctx).astype('float32')
        begin_state = model.begin_state(func=mx.nd.random.uniform, low=0, high=1,
                                        batch_size=batch, dtype='float32', ctx=ctx)
        out, out_state= model(inputs, begin_state)

        prefix = "%s/awd_lstm" % tmp_path
        model.export(prefix)
        sym_file = "%s-symbol.json" % prefix
        params_file = "%s-0000.params" % prefix
        onnx_file = "%s.onnx" % prefix

        input_shapes = [(seq_length, batch), 
                        np.shape(begin_state[0][0]), np.shape(begin_state[0][1]),
                        np.shape(begin_state[1][0]), np.shape(begin_state[1][1]),
                        np.shape(begin_state[2][0]), np.shape(begin_state[2][1])]
        input_types = [np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,
                       np.float32]
        converted_model_path = mx.onnx.export_model(sym_file, params_file, input_shapes,
                                                    input_types, onnx_file, verbose=True)

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = onnxruntime.InferenceSession(onnx_file, sess_options)

        in_tensors = [inputs, begin_state[0][0], begin_state[0][1],
                      begin_state[1][0], begin_state[1][1],
                      begin_state[2][0], begin_state[2][1]]
        input_dict = dict((sess.get_inputs()[i].name, in_tensors[i].asnumpy()) for i in range(len(in_tensors)))
        pred = sess.run(None, input_dict)

        assert_almost_equal(out, pred[6])
        assert_almost_equal(out_state[0][0], pred[0])
        assert_almost_equal(out_state[0][1], pred[1])
        assert_almost_equal(out_state[1][0], pred[2])
        assert_almost_equal(out_state[1][1], pred[3])
        assert_almost_equal(out_state[2][0], pred[4])
        assert_almost_equal(out_state[2][1], pred[5])

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.parametrize('model_name', ['ernie_12_768_12'])
@pytest.mark.parametrize('model_specific_logics', [None, 'gluonnlp_bert_uninterleaved'])
def test_ernie_inference_onnxruntime(tmp_path, model_name, model_specific_logics):
    tmp_path = str(tmp_path)
    try:
        import gluonnlp as nlp
        dataset = 'baidu_ernie_uncased'
        ctx = mx.cpu(0)
        model, vocab = nlp.model.get_model(
            name=model_name,
            ctx=ctx,
            dataset_name=dataset,
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            num_layers = 3,
            hparam_allow_override = True,
            use_classifier=False)

        model.hybridize(static_alloc=True)

        batch = 5
        seq_length = 16
        # create synthetic test data
        inputs = mx.nd.random.uniform(0, 17964, shape=(batch, seq_length), dtype='float32')
        token_types = mx.nd.random.uniform(0, 2, shape=(batch, seq_length), dtype='float32')
        valid_length = mx.nd.array([seq_length] * batch, dtype='float32')

        seq_encoding, cls_encoding = model(inputs, token_types, valid_length)

        prefix = "%s/ernie" % tmp_path
        model.export(prefix)
        sym_file = "%s-symbol.json" % prefix
        params_file = "%s-0000.params" % prefix
        onnx_file = "%s.onnx" % prefix

        input_shapes = [(batch, seq_length), (batch, seq_length), (batch,)]
        input_types = [np.float32, np.float32, np.float32]

        # this is only for when model_specific_logics='gluonnlp_bert_uninterleaved'
        cheat_sheet = {
            'qkv_hidden': 768,
            'num_heads': 12,
            'head_dim': 64
        }

        converted_model_path = mx.onnx.export_model(sym_file, params_file, input_shapes,
                                                    input_types, onnx_file,
                                                    model_specific_logics=model_specific_logics,
                                                    cheat_sheet=cheat_sheet)

        # create onnxruntime session using the generated onnx file
        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)

        seq_encoding, cls_encoding = model(inputs, token_types, valid_length)

        onnx_inputs = [inputs, token_types, valid_length]
        input_dict = dict((session.get_inputs()[i].name, onnx_inputs[i].asnumpy()) for i in range(len(onnx_inputs)))
        pred_onx, cls_onx = session.run(None, input_dict)

        assert_almost_equal(seq_encoding, pred_onx)
        assert_almost_equal(cls_encoding, cls_onx)

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.parametrize('model_name', ['transformer_en_de_512'])
def test_transformer_pretrained_inference_onnxruntime(tmp_path, model_name):
    tmp_path = str(tmp_path)
    try:
        import gluonnlp as nlp
        dataset = 'WMT2014'
        ctx = mx.cpu(0)
        model, _, _ = nlp.model.get_model(
            name=model_name,
            ctx=ctx,
            pretrained=True,
            dataset_name=dataset)

        model.hybridize(static_alloc=False)

        batch = 7
        seq_length = 16
        C_in = 512
        C_out = 512
        src = mx.nd.random.uniform(0, 36794, shape=(batch, seq_length), dtype='float32')
        step_input = mx.nd.random.uniform(0, 36794, shape=(batch,), dtype='float32')
        src_valid_length = mx.nd.array([seq_length] * batch, dtype='float32')

        encoder_outputs, encoder_additional_outputs = model.encode(src,
                                                                   valid_length=src_valid_length)

        decoder_states = model.decoder.init_state_from_encoder(encoder_outputs, src_valid_length)

        step_output, states, additional_outputs = model.decode_step(step_input, decoder_states)

        # skip export of 'decoder' as it's used for training only
        for component in ['encoder', 'one_step_ahead_decoder', 'src_embed', 'tgt_embed',
                         'tgt_proj']:

            prefix = "%s/%s" %(tmp_path, component)
            component = getattr(model, component)
            component.export(prefix)
            sym_file = "%s-symbol.json" % prefix
            params_file = "%s-0000.params" % prefix
            onnx_file = "%s.onnx" % prefix

        def export_to_onnx(prefix, input_shapes, input_types, **kwargs):
            sym_file = "%s-symbol.json" % prefix
            params_file = "%s-0000.params" % prefix
            onnx_file = "%s.onnx" % prefix
            return mx.onnx.export_model(sym_file, params_file, input_shapes, input_types,
                                        onnx_file, **kwargs)

        def onnx_runtime_predict(onnx_file, onnx_inputs):
            ses_opt = onnxruntime.SessionOptions()
            ses_opt.log_severity_level = 3
            session = onnxruntime.InferenceSession(onnx_file, ses_opt)
            input_dict = dict((session.get_inputs()[i].name, onnx_inputs[i].asnumpy())
                            for i in range(len(onnx_inputs)))
            return session.run(None, input_dict)

        def verify_encoder():
            inputs = mx.nd.random.uniform(-1, 1, shape=(batch, seq_length, C_in), dtype='float32')
            valid_length = mx.nd.array([seq_length] * batch, dtype='float32')
            pred = model.encoder(inputs, valid_length=valid_length)

            prefix = "%s/encoder" %tmp_path
            input_shapes = [(batch, seq_length, C_in), (batch,)]
            input_types = [np.float32, np.float32]
            onnx_file = export_to_onnx(prefix, input_shapes, input_types)
            onnx_inputs = [inputs, valid_length]
            pred_onx = onnx_runtime_predict(onnx_file, onnx_inputs)

            assert_almost_equal(pred[0], pred_onx[0])

        def verify_src_embed():
            src = mx.nd.random.uniform(0, 36794, shape=(batch, seq_length), dtype='float32')
            pred = model.src_embed(src)

            prefix = "%s/src_embed" %tmp_path
            input_shapes = [(batch, seq_length)]
            input_types = [np.float32]
            onnx_file = export_to_onnx(prefix, input_shapes, input_types)
            onnx_inputs = [src]
            pred_onx = onnx_runtime_predict(onnx_file, onnx_inputs)

            assert_almost_equal(pred, pred_onx[0])

        def verify_tgt_embed():
            tgt = mx.nd.random.uniform(0, 36794, shape=(batch, seq_length), dtype='float32')
            pred = model.tgt_embed(tgt)

            prefix = "%s/tgt_embed" %tmp_path
            input_shapes = [(batch, seq_length)]
            input_types = [np.float32]
            onnx_file = export_to_onnx(prefix, input_shapes, input_types)
            onnx_inputs = [tgt]
            pred_onx = onnx_runtime_predict(onnx_file, onnx_inputs)

            assert_almost_equal(pred, pred_onx[0])

        def verify_tgt_proj():
            decoder_out = mx.nd.random.uniform(0, 512, shape=(batch, seq_length, C_out),
                                               dtype='float32')
            pred = model.tgt_proj(decoder_out)

            prefix = "%s/tgt_proj" %tmp_path
            input_shapes = [(batch, seq_length, C_out)]
            input_types = [np.float32]
            onnx_file = export_to_onnx(prefix, input_shapes, input_types)
            onnx_inputs = [decoder_out]
            pred_onx = onnx_runtime_predict(onnx_file, onnx_inputs)

            assert_almost_equal(pred, pred_onx[0], rtol=1.e-04, atol=1.5e-03)

        def verify_one_step_ahead_decoder():
            prefix = "%s/one_step_ahead_decoder" %tmp_path

            # the input data order
            perm = [2, 0, 1]
            input_shapes = [(batch, seq_length, C_in), (batch, seq_length, C_out),
                            (batch, seq_length)]
            input_shapes = [input_shapes[i] for i in perm]
            dynamic_input_shapes = [(batch, 'seq_length', C_in), (batch, 'seq_length', C_out),
                                    (batch, 'seq_length')]
            dynamic_input_shapes = [dynamic_input_shapes[i] for i in perm]
            input_types = [np.float32, np.float32, np.float32]
            # do a dynamic export
            onnx_file = export_to_onnx(prefix, input_shapes, input_types, dynamic=True,
                                       dynamic_input_shapes=dynamic_input_shapes)

            # step 0
            step_input = mx.nd.random.uniform(-1, 1, shape=(batch, C_in), dtype='float32')
            # mxnet
            pred, step_states, _ = model.one_step_ahead_decoder(step_input, decoder_states)
            # onnx
            # note that we need to expand the sequence axis just like in here:
            # https://github.com/dmlc/gluon-nlp/blob/v0.10.x/src/gluonnlp/model/transformer.py#L831
            input_onx = mx.nd.expand_dims(step_input, axis=1)
            onnx_inputs = [input_onx, decoder_states[0], decoder_states[1]]
            onnx_inputs = [onnx_inputs[i] for i in perm]
            pred_onx = onnx_runtime_predict(onnx_file, onnx_inputs)

            assert_almost_equal(pred, pred_onx[0])

            # step >= 1
            for i in range(20):
                step_input = mx.nd.random.uniform(-10*i, 10*i, shape=(batch, C_in), dtype='float32')
                # mxnet
                pred, step_states, _ = model.one_step_ahead_decoder(step_input, step_states)
                # onnx
                # note that we need to concat the step_input with the previous inpus
                # just like in here:
                # https://github.com/dmlc/gluon-nlp/blob/v0.10.x/src/gluonnlp/model/transformer.py#L828
                input_onx = mx.nd.concat(input_onx, mx.nd.expand_dims(step_input, axis=1), dim=1)
                onnx_inputs = [input_onx, decoder_states[0], decoder_states[1]]
                onnx_inputs = [onnx_inputs[i] for i in perm]
                pred_onx = onnx_runtime_predict(onnx_file, onnx_inputs)

                assert_almost_equal(pred, pred_onx[0])

        verify_encoder()
        verify_src_embed()
        verify_tgt_embed()
        verify_tgt_proj()
        verify_one_step_ahead_decoder()

    finally:
        shutil.rmtree(tmp_path)


@with_seed()
@pytest.mark.parametrize('model_params', [('gpt2_117m', 24), ('gpt2_345m', 48)])
def test_gpt_pretrained_inference_onnxruntime(tmp_path, model_params):
    tmp_path = str(tmp_path)
    try:
        import gluonnlp as nlp
        import urllib.request
        from zipfile import ZipFile
        import importlib.util
        import sys

        url = 'https://nlp.gluon.ai/_downloads/77d227fbc8f1613e6802acc7253cc090/text_generation.zip'
        urllib.request.urlretrieve(url, tmp_path + 'text_generation.zip')

        with ZipFile(tmp_path + 'text_generation.zip', 'r') as zipObj:
            zipObj.extractall(tmp_path)

        # load in the text_generation module, refer to:
        # https://github.com/dmlc/gluon-nlp/tree/v0.10.x/scripts/text_generation
        spec = importlib.util.spec_from_file_location(
            'text_generation',
            tmp_path + '/text_generation/__init__.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        ctx = mx.cpu(0)
        model_name= model_params[0]
        dataset= 'openai_webtext'
        # get_model() is overridden in here:
        # https://github.com/dmlc/gluon-nlp/blob/v0.10.x/scripts/text_generation/model/__init__.py#L23
        model, _ = mod.model.get_model(
            name=model_name,
            ctx=ctx,
            pretrained=True,
            dataset_name=dataset)

        model.hybridize()

        batch = 4
        seq_length = 64
        inputs = mx.nd.random.uniform(0, 50257, shape=(batch, seq_length), dtype='float32',
                                      ctx=ctx)

        pred = model(inputs)

        prefix = "%s/%s" % (tmp_path, model_name)
        model.export(prefix)
        sym_file = "%s-symbol.json" % prefix
        params_file = "%s-0000.params" % prefix
        onnx_file = "%s.onnx" % prefix

        input_shapes = [(batch, seq_length)]
        input_types = [np.float32]
        converted_model_path = mx.onnx.export_model(sym_file, params_file, input_shapes,
                                                    input_types, onnx_file)

        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)
        onnx_inputs = [inputs]
        input_dict = dict((session.get_inputs()[i].name, onnx_inputs[i].asnumpy()) for i in range(len(onnx_inputs)))
        pred_onx = session.run(None, input_dict)

        # check output
        assert_almost_equal(pred[0], pred_onx[0])
        # check states
        num_states = model_params[1]
        for i in range(num_states):
            assert_almost_equal(pred[1][i], pred_onx[i+1])

    finally:
        shutil.rmtree(tmp_path)
