# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8

from collections import Counter

from common import assertRaises
import mxnet as mx
from mxnet import ndarray as nd
from mxnet.test_utils import *
from mxnet.contrib import text

import pytest

mx.reset_np()

def _get_test_str_of_tokens(token_delim, seq_delim):
    seq1 = token_delim + token_delim.join(['Life', 'is', 'great', '!']) + token_delim + seq_delim
    seq2 = token_delim + token_delim.join(['life', 'is', 'good', '.']) + token_delim + seq_delim
    seq3 = token_delim + token_delim.join(['life', "isn't", 'bad', '.']) + token_delim + seq_delim
    seqs = seq1 + seq2 + seq3
    return seqs


def _test_count_tokens_from_str_with_delims(token_delim, seq_delim):
    source_str = _get_test_str_of_tokens(token_delim, seq_delim)

    cnt1 = text.utils.count_tokens_from_str(
        source_str, token_delim, seq_delim, to_lower=False)
    assert cnt1 == Counter(
        {'is': 2, 'life': 2, '.': 2, 'Life': 1, 'great': 1, '!': 1, 'good': 1, "isn't": 1,
         'bad': 1})

    cnt2 = text.utils.count_tokens_from_str(
        source_str, token_delim, seq_delim, to_lower=True)
    assert cnt2 == Counter(
        {'life': 3, 'is': 2, '.': 2, 'great': 1, '!': 1, 'good': 1, "isn't": 1, 'bad': 1})

    counter_to_update = Counter({'life': 2})

    cnt3 = text.utils.count_tokens_from_str(
        source_str, token_delim, seq_delim, to_lower=False,
        counter_to_update=counter_to_update.copy())
    assert cnt3 == Counter(
        {'is': 2, 'life': 4, '.': 2, 'Life': 1, 'great': 1, '!': 1, 'good': 1, "isn't": 1,
         'bad': 1})

    cnt4 = text.utils.count_tokens_from_str(
        source_str, token_delim, seq_delim, to_lower=True,
        counter_to_update=counter_to_update.copy())
    assert cnt4 == Counter(
        {'life': 5, 'is': 2, '.': 2, 'great': 1, '!': 1, 'good': 1, "isn't": 1, 'bad': 1})


def test_count_tokens_from_str():
    _test_count_tokens_from_str_with_delims(' ', '\n')
    _test_count_tokens_from_str_with_delims('IS', 'LIFE')


def test_tokens_to_indices():
    counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])

    vocab = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=1, unknown_token='<unk>',
                                  reserved_tokens=None)

    i1 = vocab.to_indices('c')
    assert i1 == 1

    i2 = vocab.to_indices(['c'])
    assert i2 == [1]

    i3 = vocab.to_indices(['<unk>', 'non-exist'])
    assert i3 == [0, 0]

    i4 = vocab.to_indices(['a', 'non-exist', 'a', 'b'])
    assert i4 == [3, 0, 3, 2]


def test_indices_to_tokens():
    counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])

    vocab = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=1,
                                  unknown_token='<unknown>', reserved_tokens=None)
    i1 = vocab.to_tokens(1)
    assert i1 == 'c'

    i2 = vocab.to_tokens([1])
    assert i2 == ['c']

    i3 = vocab.to_tokens([0, 0])
    assert i3 == ['<unknown>', '<unknown>']

    i4 = vocab.to_tokens([3, 0, 3, 2])
    assert i4 == ['a', '<unknown>', 'a', 'b']

    assertRaises(ValueError, vocab.to_tokens, 100)


def test_download_embed():
    @text.embedding.register
    class Test(text.embedding._TokenEmbedding):
        # 33 bytes.
        pretrained_file_name_sha1 = \
            {'embedding_test.vec': '29b9a6511cf4b5aae293c44a9ec1365b74f2a2f8'}
        namespace = 'test'

        def __init__(self, embedding_root='embeddings', init_unknown_vec=nd.zeros, **kwargs):
            pretrained_file_name = 'embedding_test.vec'
            Test._check_pretrained_file_names(pretrained_file_name)

            super(Test, self).__init__(**kwargs)

            pretrained_file_path = Test._get_pretrained_file(embedding_root, pretrained_file_name)

            self._load_embedding(pretrained_file_path, ' ', init_unknown_vec)

    test_embed = text.embedding.create('test')
    assert test_embed.token_to_idx['hello'] == 1
    assert test_embed.token_to_idx['world'] == 2
    assert_almost_equal(test_embed.idx_to_vec[1].asnumpy(), (nd.arange(5) + 1).asnumpy())
    assert_almost_equal(test_embed.idx_to_vec[2].asnumpy(), (nd.arange(5) + 6).asnumpy())
    assert_almost_equal(test_embed.idx_to_vec[0].asnumpy(), nd.zeros((5,)).asnumpy())


def _mk_my_pretrain_file(path, token_delim, pretrain_file):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.1', '0.2', '0.3', '0.4', '0.5']) + '\n'
    seq2 = token_delim.join(['b', '0.6', '0.7', '0.8', '0.9', '1.0']) + '\n'
    seqs = seq1 + seq2
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def _mk_my_pretrain_file2(path, token_delim, pretrain_file):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.01', '0.02', '0.03', '0.04', '0.05']) + '\n'
    seq2 = token_delim.join(['c', '0.06', '0.07', '0.08', '0.09', '0.1']) + '\n'
    seqs = seq1 + seq2
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def _mk_my_pretrain_file3(path, token_delim, pretrain_file):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.1', '0.2', '0.3', '0.4', '0.5']) + '\n'
    seq2 = token_delim.join(['b', '0.6', '0.7', '0.8', '0.9', '1.0']) + '\n'
    seq3 = token_delim.join(['<unk1>', '1.1', '1.2', '1.3', '1.4',
                             '1.5']) + '\n'
    seqs = seq1 + seq2 + seq3
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def _mk_my_pretrain_file4(path, token_delim, pretrain_file):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.01', '0.02', '0.03', '0.04', '0.05']) + '\n'
    seq2 = token_delim.join(['c', '0.06', '0.07', '0.08', '0.09', '0.1']) + '\n'
    seq3 = token_delim.join(['<unk2>', '0.11', '0.12', '0.13', '0.14', '0.15']) + '\n'
    seqs = seq1 + seq2 + seq3
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def _mk_my_invalid_pretrain_file(path, token_delim, pretrain_file):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.1', '0.2', '0.3', '0.4', '0.5']) + '\n'
    seq2 = token_delim.join(['b', '0.6', '0.7', '0.8', '0.9', '1.0']) + '\n'
    seq3 = token_delim.join(['c']) + '\n'
    seqs = seq1 + seq2 + seq3
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def _mk_my_invalid_pretrain_file2(path, token_delim, pretrain_file):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.1', '0.2', '0.3', '0.4', '0.5']) + '\n'
    seq2 = token_delim.join(['b', '0.6', '0.7', '0.8', '0.9', '1.0']) + '\n'
    seq3 = token_delim.join(['c', '0.6', '0.7', '0.8']) + '\n'
    seqs = seq1 + seq2 + seq3
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def test_custom_embed():
    embed_root = 'embeddings'
    embed_name = 'my_embed'
    elem_delim = '\t'
    pretrain_file = 'my_pretrain_file.txt'

    _mk_my_pretrain_file(os.path.join(embed_root, embed_name), elem_delim, pretrain_file)

    pretrain_file_path = os.path.join(embed_root, embed_name, pretrain_file)

    my_embed = text.embedding.CustomEmbedding(pretrain_file_path, elem_delim)

    assert len(my_embed) == 3
    assert my_embed.vec_len == 5
    assert my_embed.token_to_idx['a'] == 1
    assert my_embed.idx_to_token[1] == 'a'

    first_vec = my_embed.idx_to_vec[0]
    assert_almost_equal(first_vec.asnumpy(), np.array([0, 0, 0, 0, 0]))

    unk_vec = my_embed.get_vecs_by_tokens('A')
    assert_almost_equal(unk_vec.asnumpy(), np.array([0, 0, 0, 0, 0]))

    a_vec = my_embed.get_vecs_by_tokens('A', lower_case_backup=True)
    assert_almost_equal(a_vec.asnumpy(), np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

    unk_vecs = my_embed.get_vecs_by_tokens(['<unk$unk@unk>', '<unk$unk@unk>'])
    assert_almost_equal(unk_vecs.asnumpy(), np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))

    # Test loaded unknown vectors.
    pretrain_file2 = 'my_pretrain_file2.txt'
    _mk_my_pretrain_file3(os.path.join(embed_root, embed_name), elem_delim, pretrain_file2)
    pretrain_file_path = os.path.join(embed_root, embed_name, pretrain_file2)
    my_embed2 = text.embedding.CustomEmbedding(pretrain_file_path, elem_delim,
                                               init_unknown_vec=nd.ones, unknown_token='<unk>')
    unk_vec2 = my_embed2.get_vecs_by_tokens('<unk>')
    assert_almost_equal(unk_vec2.asnumpy(), np.array([1, 1, 1, 1, 1]))
    unk_vec2 = my_embed2.get_vecs_by_tokens('<unk$unk@unk>')
    assert_almost_equal(unk_vec2.asnumpy(), np.array([1, 1, 1, 1, 1]))

    my_embed3 = text.embedding.CustomEmbedding(pretrain_file_path, elem_delim,
                                               init_unknown_vec=nd.ones, unknown_token='<unk1>')
    unk_vec3 = my_embed3.get_vecs_by_tokens('<unk1>')
    assert_almost_equal(unk_vec3.asnumpy(), np.array([1.1, 1.2, 1.3, 1.4, 1.5]))
    unk_vec3 = my_embed3.get_vecs_by_tokens('<unk$unk@unk>')
    assert_almost_equal(unk_vec3.asnumpy(), np.array([1.1, 1.2, 1.3, 1.4, 1.5]))

    # Test error handling.
    invalid_pretrain_file = 'invalid_pretrain_file.txt'
    _mk_my_invalid_pretrain_file(os.path.join(embed_root, embed_name), elem_delim,
                                 invalid_pretrain_file)
    pretrain_file_path = os.path.join(embed_root, embed_name, invalid_pretrain_file)
    assertRaises(AssertionError, text.embedding.CustomEmbedding, pretrain_file_path, elem_delim)

    invalid_pretrain_file2 = 'invalid_pretrain_file2.txt'
    _mk_my_invalid_pretrain_file2(os.path.join(embed_root, embed_name), elem_delim,
                                  invalid_pretrain_file2)
    pretrain_file_path = os.path.join(embed_root, embed_name, invalid_pretrain_file2)
    assertRaises(AssertionError, text.embedding.CustomEmbedding, pretrain_file_path, elem_delim)


def test_vocabulary():
    counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])

    v1 = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=1, unknown_token='<unk>',
                               reserved_tokens=None)
    assert len(v1) == 5
    assert v1.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3, 'some_word$': 4}
    assert v1.idx_to_token[1] == 'c'
    assert v1.unknown_token == '<unk>'
    assert v1.reserved_tokens is None

    v2 = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=2, unknown_token='<unk>',
                               reserved_tokens=None)
    assert len(v2) == 3
    assert v2.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2}
    assert v2.idx_to_token[1] == 'c'
    assert v2.unknown_token == '<unk>'
    assert v2.reserved_tokens is None

    v3 = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=100, unknown_token='<unk>',
                               reserved_tokens=None)
    assert len(v3) == 1
    assert v3.token_to_idx == {'<unk>': 0}
    assert v3.idx_to_token[0] == '<unk>'
    assert v3.unknown_token == '<unk>'
    assert v3.reserved_tokens is None

    v4 = text.vocab.Vocabulary(counter, most_freq_count=2, min_freq=1, unknown_token='<unk>',
                               reserved_tokens=None)
    assert len(v4) == 3
    assert v4.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2}
    assert v4.idx_to_token[1] == 'c'
    assert v4.unknown_token == '<unk>'
    assert v4.reserved_tokens is None

    v5 = text.vocab.Vocabulary(counter, most_freq_count=3, min_freq=1, unknown_token='<unk>',
                               reserved_tokens=None)
    assert len(v5) == 4
    assert v5.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3}
    assert v5.idx_to_token[1] == 'c'
    assert v5.unknown_token == '<unk>'
    assert v5.reserved_tokens is None

    v6 = text.vocab.Vocabulary(counter, most_freq_count=100, min_freq=1, unknown_token='<unk>',
                               reserved_tokens=None)
    assert len(v6) == 5
    assert v6.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3,
                               'some_word$': 4}
    assert v6.idx_to_token[1] == 'c'
    assert v6.unknown_token == '<unk>'
    assert v6.reserved_tokens is None

    v7 = text.vocab.Vocabulary(counter, most_freq_count=1, min_freq=2, unknown_token='<unk>',
                               reserved_tokens=None)
    assert len(v7) == 2
    assert v7.token_to_idx == {'<unk>': 0, 'c': 1}
    assert v7.idx_to_token[1] == 'c'
    assert v7.unknown_token == '<unk>'
    assert v7.reserved_tokens is None

    assertRaises(AssertionError, text.vocab.Vocabulary, counter, most_freq_count=None,
                 min_freq=0, unknown_token='<unknown>', reserved_tokens=['b'])

    assertRaises(AssertionError, text.vocab.Vocabulary, counter, most_freq_count=None,
                 min_freq=1, unknown_token='<unknown>', reserved_tokens=['b', 'b'])

    assertRaises(AssertionError, text.vocab.Vocabulary, counter, most_freq_count=None,
                 min_freq=1, unknown_token='<unknown>', reserved_tokens=['b', '<unknown>'])

    v8 = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=1, unknown_token='<unknown>',
                               reserved_tokens=['b'])
    assert len(v8) == 5
    assert v8.token_to_idx == {'<unknown>': 0, 'b': 1, 'c': 2, 'a': 3, 'some_word$': 4}
    assert v8.idx_to_token[1] == 'b'
    assert v8.unknown_token == '<unknown>'
    assert v8.reserved_tokens == ['b']

    v9 = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=2, unknown_token='<unk>',
                               reserved_tokens=['b', 'a'])
    assert len(v9) == 4
    assert v9.token_to_idx == {'<unk>': 0, 'b': 1, 'a': 2, 'c': 3}
    assert v9.idx_to_token[1] == 'b'
    assert v9.unknown_token == '<unk>'
    assert v9.reserved_tokens == ['b', 'a']

    v10 = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=100, unknown_token='<unk>',
                                reserved_tokens=['b', 'c'])
    assert len(v10) == 3
    assert v10.token_to_idx == {'<unk>': 0, 'b': 1, 'c': 2}
    assert v10.idx_to_token[1] == 'b'
    assert v10.unknown_token == '<unk>'
    assert v10.reserved_tokens == ['b', 'c']

    v11 = text.vocab.Vocabulary(counter, most_freq_count=1, min_freq=2, unknown_token='<unk>',
                                reserved_tokens=['<pad>', 'b'])
    assert len(v11) == 4
    assert v11.token_to_idx == {'<unk>': 0, '<pad>': 1, 'b': 2, 'c': 3}
    assert v11.idx_to_token[1] == '<pad>'
    assert v11.unknown_token == '<unk>'
    assert v11.reserved_tokens == ['<pad>', 'b']

    v12 = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=2, unknown_token='b',
                                reserved_tokens=['<pad>'])
    assert len(v12) == 3
    assert v12.token_to_idx == {'b': 0, '<pad>': 1, 'c': 2}
    assert v12.idx_to_token[1] == '<pad>'
    assert v12.unknown_token == 'b'
    assert v12.reserved_tokens == ['<pad>']

    v13 = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=2, unknown_token='a',
                                reserved_tokens=['<pad>'])
    assert len(v13) == 4
    assert v13.token_to_idx == {'a': 0, '<pad>': 1, 'c': 2, 'b': 3}
    assert v13.idx_to_token[1] == '<pad>'
    assert v13.unknown_token == 'a'
    assert v13.reserved_tokens == ['<pad>']

    counter_tuple = Counter([('a', 'a'), ('b', 'b'), ('b', 'b'), ('c', 'c'), ('c', 'c'), ('c', 'c'),
                             ('some_word$', 'some_word$')])

    v14 = text.vocab.Vocabulary(counter_tuple, most_freq_count=None, min_freq=1,
                                unknown_token=('<unk>', '<unk>'), reserved_tokens=None)
    assert len(v14) == 5
    assert v14.token_to_idx == {('<unk>', '<unk>'): 0, ('c', 'c'): 1, ('b', 'b'): 2, ('a', 'a'): 3,
                                ('some_word$', 'some_word$'): 4}
    assert v14.idx_to_token[1] == ('c', 'c')
    assert v14.unknown_token == ('<unk>', '<unk>')
    assert v14.reserved_tokens is None


def test_custom_embedding_with_vocabulary():
    embed_root = 'embeddings'
    embed_name = 'my_embed'
    elem_delim = '\t'
    pretrain_file = 'my_pretrain_file1.txt'

    _mk_my_pretrain_file(os.path.join(embed_root, embed_name), elem_delim, pretrain_file)

    pretrain_file_path = os.path.join(embed_root, embed_name, pretrain_file)

    counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])

    v1 = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=1, unknown_token='<unk>',
                               reserved_tokens=['<pad>'])

    e1 = text.embedding.CustomEmbedding(pretrain_file_path, elem_delim, init_unknown_vec=nd.ones,
                                        vocabulary=v1)

    assert e1.token_to_idx == {'<unk>': 0, '<pad>': 1, 'c': 2, 'b': 3, 'a': 4, 'some_word$': 5}
    assert e1.idx_to_token == ['<unk>', '<pad>', 'c', 'b', 'a', 'some_word$']

    assert_almost_equal(e1.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [0.6, 0.7, 0.8, 0.9, 1],
                                  [0.1, 0.2, 0.3, 0.4, 0.5],
                                  [1, 1, 1, 1, 1]])
                        )

    assert e1.vec_len == 5
    assert e1.reserved_tokens == ['<pad>']

    assert_almost_equal(e1.get_vecs_by_tokens('c').asnumpy(),
                        np.array([1, 1, 1, 1, 1])
                        )

    assert_almost_equal(e1.get_vecs_by_tokens(['c']).asnumpy(),
                        np.array([[1, 1, 1, 1, 1]])
                        )

    assert_almost_equal(e1.get_vecs_by_tokens(['a', 'not_exist']).asnumpy(),
                        np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                                  [1, 1, 1, 1, 1]])
                        )

    assert_almost_equal(e1.get_vecs_by_tokens(['a', 'b']).asnumpy(),
                        np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                                  [0.6, 0.7, 0.8, 0.9, 1]])
                        )

    assert_almost_equal(e1.get_vecs_by_tokens(['A', 'b']).asnumpy(),
                        np.array([[1, 1, 1, 1, 1],
                                  [0.6, 0.7, 0.8, 0.9, 1]])
                        )

    assert_almost_equal(e1.get_vecs_by_tokens(['A', 'b'], lower_case_backup=True).asnumpy(),
                        np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                                  [0.6, 0.7, 0.8, 0.9, 1]])
                        )

    e1.update_token_vectors(['a', 'b'],
                            nd.array([[2, 2, 2, 2, 2],
                                      [3, 3, 3, 3, 3]])
                            )

    assert_almost_equal(e1.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )

    assertRaises(ValueError, e1.update_token_vectors, 'unknown$$$', nd.array([0, 0, 0, 0, 0]))

    assertRaises(AssertionError, e1.update_token_vectors, '<unk>',
                 nd.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))

    assertRaises(AssertionError, e1.update_token_vectors, '<unk>', nd.array([0]))

    e1.update_token_vectors(['<unk>'], nd.array([0, 0, 0, 0, 0]))
    assert_almost_equal(e1.idx_to_vec.asnumpy(),
                        np.array([[0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )
    e1.update_token_vectors(['<unk>'], nd.array([[10, 10, 10, 10, 10]]))
    assert_almost_equal(e1.idx_to_vec.asnumpy(),
                        np.array([[10, 10, 10, 10, 10],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )
    e1.update_token_vectors('<unk>', nd.array([0, 0, 0, 0, 0]))
    assert_almost_equal(e1.idx_to_vec.asnumpy(),
                        np.array([[0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )
    e1.update_token_vectors('<unk>', nd.array([[10, 10, 10, 10, 10]]))
    assert_almost_equal(e1.idx_to_vec.asnumpy(),
                        np.array([[10, 10, 10, 10, 10],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )


@pytest.mark.skip(reason='https://github.com/apache/incubator-mxnet/issues/18282')
def test_composite_embedding_with_one_embedding():
    embed_root = 'embeddings'
    embed_name = 'my_embed'
    elem_delim = '\t'
    pretrain_file = 'my_pretrain_file1.txt'

    _mk_my_pretrain_file(os.path.join(embed_root, embed_name), elem_delim, pretrain_file)

    pretrain_file_path = os.path.join(embed_root, embed_name, pretrain_file)

    my_embed = text.embedding.CustomEmbedding(pretrain_file_path, elem_delim,
                                              init_unknown_vec=nd.ones)

    counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])

    v1 = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=1, unknown_token='<unk>',
                               reserved_tokens=['<pad>'])
    ce1 = text.embedding.CompositeEmbedding(v1, my_embed)

    assert ce1.token_to_idx == {'<unk>': 0, '<pad>': 1, 'c': 2, 'b': 3, 'a': 4, 'some_word$': 5}
    assert ce1.idx_to_token == ['<unk>', '<pad>', 'c', 'b', 'a', 'some_word$']

    assert_almost_equal(ce1.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [0.6, 0.7, 0.8, 0.9, 1],
                                  [0.1, 0.2, 0.3, 0.4, 0.5],
                                  [1, 1, 1, 1, 1]])
                        )

    assert ce1.vec_len == 5
    assert ce1.reserved_tokens == ['<pad>']

    assert_almost_equal(ce1.get_vecs_by_tokens('c').asnumpy(),
                        np.array([1, 1, 1, 1, 1])
                        )

    assert_almost_equal(ce1.get_vecs_by_tokens(['c']).asnumpy(),
                        np.array([[1, 1, 1, 1, 1]])
                        )

    assert_almost_equal(ce1.get_vecs_by_tokens(['a', 'not_exist']).asnumpy(),
                        np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                                  [1, 1, 1, 1, 1]])
                        )

    assert_almost_equal(ce1.get_vecs_by_tokens(['a', 'b']).asnumpy(),
                        np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                                  [0.6, 0.7, 0.8, 0.9, 1]])
                        )

    assert_almost_equal(ce1.get_vecs_by_tokens(['A', 'b']).asnumpy(),
                        np.array([[1, 1, 1, 1, 1],
                                  [0.6, 0.7, 0.8, 0.9, 1]])
                        )

    assert_almost_equal(ce1.get_vecs_by_tokens(['A', 'b'], lower_case_backup=True).asnumpy(),
                        np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                                  [0.6, 0.7, 0.8, 0.9, 1]])
                        )

    ce1.update_token_vectors(['a', 'b'],
                             nd.array([[2, 2, 2, 2, 2],
                                      [3, 3, 3, 3, 3]])
                             )

    assert_almost_equal(ce1.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )

    assertRaises(ValueError, ce1.update_token_vectors, 'unknown$$$', nd.array([0, 0, 0, 0, 0]))

    assertRaises(AssertionError, ce1.update_token_vectors, '<unk>',
                 nd.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))

    assertRaises(AssertionError, ce1.update_token_vectors, '<unk>', nd.array([0]))

    ce1.update_token_vectors(['<unk>'], nd.array([0, 0, 0, 0, 0]))
    assert_almost_equal(ce1.idx_to_vec.asnumpy(),
                        np.array([[0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )
    ce1.update_token_vectors(['<unk>'], nd.array([[10, 10, 10, 10, 10]]))
    assert_almost_equal(ce1.idx_to_vec.asnumpy(),
                        np.array([[10, 10, 10, 10, 10],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )
    ce1.update_token_vectors('<unk>', nd.array([0, 0, 0, 0, 0]))
    assert_almost_equal(ce1.idx_to_vec.asnumpy(),
                        np.array([[0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )
    ce1.update_token_vectors('<unk>', nd.array([[10, 10, 10, 10, 10]]))
    assert_almost_equal(ce1.idx_to_vec.asnumpy(),
                        np.array([[10, 10, 10, 10, 10],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )


def test_composite_embedding_with_two_embeddings():
    embed_root = '.'
    embed_name = 'my_embed'
    elem_delim = '\t'
    pretrain_file1 = 'my_pretrain_file1.txt'
    pretrain_file2 = 'my_pretrain_file2.txt'

    _mk_my_pretrain_file(os.path.join(embed_root, embed_name), elem_delim, pretrain_file1)
    _mk_my_pretrain_file2(os.path.join(embed_root, embed_name), elem_delim, pretrain_file2)

    pretrain_file_path1 = os.path.join(embed_root, embed_name, pretrain_file1)
    pretrain_file_path2 = os.path.join(embed_root, embed_name, pretrain_file2)

    my_embed1 = text.embedding.CustomEmbedding(pretrain_file_path1, elem_delim,
                                               init_unknown_vec=nd.ones)
    my_embed2 = text.embedding.CustomEmbedding(pretrain_file_path2, elem_delim)

    counter = Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])

    v1 = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=1, unknown_token='<unk>',
                               reserved_tokens=None)
    ce1 = text.embedding.CompositeEmbedding(v1, [my_embed1, my_embed2])

    assert ce1.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3, 'some_word$': 4}
    assert ce1.idx_to_token == ['<unk>', 'c', 'b', 'a', 'some_word$']

    assert_almost_equal(ce1.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 0.06, 0.07, 0.08, 0.09, 0.1],
                                  [0.6, 0.7, 0.8, 0.9, 1, 0, 0, 0, 0, 0],
                                  [0.1, 0.2, 0.3, 0.4, 0.5,
                                   0.01, 0.02, 0.03, 0.04, 0.05],
                                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
                        )

    assert ce1.vec_len == 10
    assert ce1.reserved_tokens is None
    assert_almost_equal(ce1.get_vecs_by_tokens('c').asnumpy(),
                        np.array([1, 1, 1, 1, 1, 0.06, 0.07, 0.08, 0.09, 0.1])
                        )

    assert_almost_equal(ce1.get_vecs_by_tokens(['b', 'not_exist']).asnumpy(),
                        np.array([[0.6, 0.7, 0.8, 0.9, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
                        )

    ce1.update_token_vectors(['a', 'b'],
                            nd.array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])
                            )
    assert_almost_equal(ce1.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 0.06, 0.07, 0.08, 0.09, 0.1],
                                  [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
                        )

    # Test loaded unknown tokens
    pretrain_file3 = 'my_pretrain_file3.txt'
    pretrain_file4 = 'my_pretrain_file4.txt'

    _mk_my_pretrain_file3(os.path.join(embed_root, embed_name), elem_delim, pretrain_file3)
    _mk_my_pretrain_file4(os.path.join(embed_root, embed_name), elem_delim, pretrain_file4)

    pretrain_file_path3 = os.path.join(embed_root, embed_name, pretrain_file3)
    pretrain_file_path4 = os.path.join(embed_root, embed_name, pretrain_file4)

    my_embed3 = text.embedding.CustomEmbedding(pretrain_file_path3, elem_delim,
                                               init_unknown_vec=nd.ones, unknown_token='<unk1>')
    my_embed4 = text.embedding.CustomEmbedding(pretrain_file_path4, elem_delim,
                                               unknown_token='<unk2>')

    v2 = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=1, unknown_token='<unk>',
                               reserved_tokens=None)
    ce2 = text.embedding.CompositeEmbedding(v2, [my_embed3, my_embed4])
    assert_almost_equal(ce2.idx_to_vec.asnumpy(),
                        np.array([[1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.06, 0.07, 0.08, 0.09, 0.1],
                                  [0.6, 0.7, 0.8, 0.9, 1,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [0.1, 0.2, 0.3, 0.4, 0.5,
                                   0.01, 0.02, 0.03, 0.04, 0.05],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15]])
                        )

    v3 = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=1, unknown_token='<unk1>',
                               reserved_tokens=None)
    ce3 = text.embedding.CompositeEmbedding(v3, [my_embed3, my_embed4])
    assert_almost_equal(ce3.idx_to_vec.asnumpy(),
                        np.array([[1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.06, 0.07, 0.08, 0.09, 0.1],
                                  [0.6, 0.7, 0.8, 0.9, 1,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [0.1, 0.2, 0.3, 0.4, 0.5,
                                   0.01, 0.02, 0.03, 0.04, 0.05],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15]])
                        )

    v4 = text.vocab.Vocabulary(counter, most_freq_count=None, min_freq=1, unknown_token='<unk2>',
                               reserved_tokens=None)
    ce4 = text.embedding.CompositeEmbedding(v4, [my_embed3, my_embed4])
    assert_almost_equal(ce4.idx_to_vec.asnumpy(),
                        np.array([[1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.06, 0.07, 0.08, 0.09, 0.1],
                                  [0.6, 0.7, 0.8, 0.9, 1,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [0.1, 0.2, 0.3, 0.4, 0.5,
                                   0.01, 0.02, 0.03, 0.04, 0.05],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15]])
                        )

    counter2 = Counter(['b', 'b', 'c', 'c', 'c', 'some_word$'])

    v5 = text.vocab.Vocabulary(counter2, most_freq_count=None, min_freq=1, unknown_token='a',
                               reserved_tokens=None)
    ce5 = text.embedding.CompositeEmbedding(v5, [my_embed3, my_embed4])
    assert ce5.token_to_idx == {'a': 0, 'c': 1, 'b': 2, 'some_word$': 3}
    assert ce5.idx_to_token == ['a', 'c', 'b', 'some_word$']
    assert_almost_equal(ce5.idx_to_vec.asnumpy(),
                        np.array([[1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.06, 0.07, 0.08, 0.09, 0.1],
                                  [0.6, 0.7, 0.8, 0.9, 1,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15]])
                        )


def test_get_and_pretrain_file_names():
    assert len(text.embedding.get_pretrained_file_names(
        embedding_name='fasttext')) == 327

    assert len(text.embedding.get_pretrained_file_names(embedding_name='glove')) == 10

    reg = text.embedding.get_pretrained_file_names(embedding_name=None)

    assert len(reg['glove']) == 10
    assert len(reg['fasttext']) == 327

    assertRaises(KeyError, text.embedding.get_pretrained_file_names, 'unknown$$')

