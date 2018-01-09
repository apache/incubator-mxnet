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

# coding: utf-8
# pylint: disable=consider-iterating-dictionary

"""Text token embeddings."""
from __future__ import absolute_import
from __future__ import print_function

import io
import logging
import os
import tarfile
import warnings
import zipfile

from . import constants as C
from ..gluon.utils import download
from .indexer import TokenIndexer
from .. import ndarray as nd
from .. import registry


class TokenEmbedding(TokenIndexer):
    """Token embedding base class.


    To load token embeddings from an externally hosted pre-trained
    token embedding file, such as those of GloVe and FastText, use
    `TokenEmbedding.create(embedding_name, pretrained_file_name)`. To get all
    the available `embedding_name` and `pretrained_file_name`, use
    `TokenEmbedding.get_embedding_and_pretrained_file_names()`.

    Alternatively, to load embedding vectors from a custom pre-trained token
    embedding file, use :func:`~mxnet.text.embeddings.CustomEmbedding`.

    For every unknown token, if its representation `self.unknown_token` is
    encountered in the pre-trained token embedding file, index 0 of
    `self.idx_to_vec` maps to the pre-trained token embedding vector loaded from
    the file; otherwise, index 0 of `self.idx_to_vec` maps to the token
    embedding vector initialized by `init_unknown_vec`.

    If a token is encountered multiple times in the pre-trained token embedding
    file, only the first-encountered token embedding vector will be loaded and
    the rest will be skipped.

    For the same token, its index and embedding vector may vary across different
    instances of :func:`~mxnet.text.embedding.TokenEmbedding`.


    Properties
    ----------
    vec_len : int
        The length of the embedding vector for each token.
    idx_to_vec : mxnet.ndarray.NDArray
        For all the indexed tokens in this embedding, this NDArray maps each
        token's index to an embedding vector. The largest valid index maps
        to the initialized embedding vector for every reserved token, such as an
        unknown_token token and a padding token.
    """

    def __init__(self, **kwargs):
        super(TokenEmbedding, self).__init__(**kwargs)

    @classmethod
    def _get_pretrained_file_path_from_url(cls, url, embedding_root,
                                           pretrained_file_name):
        """Get the local path to the pre-trained token embedding file from url.


        The pre-trained embedding file will be downloaded from url if it has not
        been downloaded yet or the existing file fails to match its expected
        SHA-1 hash.
        """

        embedding_cls = cls.__name__.lower()
        embedding_root = os.path.expanduser(embedding_root)

        embedding_dir = os.path.join(embedding_root, embedding_cls)
        pretrained_file_path = os.path.join(embedding_dir, pretrained_file_name)
        downloaded_file = os.path.basename(url)
        downloaded_file_path = os.path.join(embedding_dir, downloaded_file)

        expected_file_hash = cls.pretrained_file_name_sha1[pretrained_file_name]

        if hasattr(cls, 'pretrained_archive_name_sha1'):
            expected_downloaded_hash = \
                cls.pretrained_archive_name_sha1[downloaded_file]
        else:
            expected_downloaded_hash = expected_file_hash

        # If downloaded_file_path exists and matches expected_downloaded_hash,
        # there is no need to download.
        download(url, downloaded_file_path, sha1_hash=expected_downloaded_hash)

        ext = os.path.splitext(downloaded_file)[1]
        if ext == '.zip':
            with zipfile.ZipFile(downloaded_file_path, 'r') as zf:
                zf.extractall(embedding_dir)
        elif ext == '.gz':
            with tarfile.open(downloaded_file_path, 'r:gz') as tar:
                tar.extractall(path=embedding_dir)
        return pretrained_file_path

    def _load_embedding(self, pretrained_file_path, elem_delim,
                        init_unknown_vec, encoding='utf8'):
        """Load embedding vectors from the pre-trained token embedding file.


        For every unknown token, if its representation `self.unknown_token` is
        encountered in the pre-trained token embedding file, index 0 of
        `self.idx_to_vec` maps to the pre-trained token embedding vector loaded
        from the file; otherwise, index 0 of `self.idx_to_vec` maps to the text
        embedding vector initialized by `init_unknown_vec`.

        If a token is encountered multiple times in the pre-trained text
        embedding file, only the first-encountered token embedding vector will
        be loaded and the rest will be skipped.
        """

        pretrained_file_path = os.path.expanduser(pretrained_file_path)

        if not os.path.isfile(pretrained_file_path):
            raise ValueError('`pretrained_file_path` must be a valid path to '
                             'the pre-trained token embedding file.')

        with io.open(pretrained_file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()

        logging.info('Loading pre-trained token embedding vectors from %s',
                     pretrained_file_path)

        vec_len = None
        all_elems = []
        tokens = set()
        loaded_unknown_vec = None
        line_num = 0
        for line in lines:
            line_num += 1
            elems = line.rstrip().split(elem_delim)

            assert len(elems) > 1, 'At line %d of the pre-trained text ' \
                                   'embedding file: the data format of the ' \
                                   'pre-trained token embedding file %s is ' \
                                   'unexpected.' \
                                   % (line_num, pretrained_file_path)

            token, elems = elems[0], [float(i) for i in elems[1:]]

            if token == self.unknown_token and loaded_unknown_vec is None:
                loaded_unknown_vec = elems
                tokens.add(self.unknown_token)
            elif token in tokens:
                warnings.warn('At line %d of the pre-trained token embedding '
                              'file: the embedding vector for token %s has '
                              'been loaded and a duplicate embedding for the '
                              'same token is seen and skipped.'
                              % (line_num, token))
            elif len(elems) == 1:
                warnings.warn('At line %d of the pre-trained text '
                              'embedding file: token %s with 1-dimensional '
                              'vector %s is likely a header and is '
                              'skipped.' % (line_num, token, elems))
            else:
                if vec_len is None:
                    vec_len = len(elems)
                    # Reserve a vector slot for the unknown token at the
                    # very beggining because the unknown index is 0.
                    all_elems.extend([0] * vec_len)
                else:
                    assert len(elems) == vec_len, \
                        'At line %d of the pre-trained token embedding ' \
                        'file: the dimension of token %s is %d but the ' \
                        'dimension of previous tokens is %d. Dimensions ' \
                        'of all the tokens must be the same.' \
                        % (line_num, token, len(elems), vec_len)
                all_elems.extend(elems)
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1
                tokens.add(token)

        self._vec_len = vec_len
        self._idx_to_vec = nd.array(all_elems).reshape((-1, self.vec_len))

        if loaded_unknown_vec is None:
            self._idx_to_vec[C.UNKNOWN_IDX] = init_unknown_vec(
                shape=self.vec_len)
        else:
            self._idx_to_vec[C.UNKNOWN_IDX] = nd.array(loaded_unknown_vec)

    @property
    def vec_len(self):
        return self._vec_len

    @property
    def idx_to_vec(self):
        return self._idx_to_vec

    def get_vecs_by_tokens(self, tokens, lower_case_backup=False):
        """Look up embedding vectors of tokens.


        Parameters
        ----------
        tokens : str or list of strs
            A token or a list of tokens.
        lower_case_backup : bool, default False
            If False, each token in the original case will be looked up; if
            True, each token in the original case will be looked up first, if
            not found in the keys of the property `token_to_idx`, the token
            in the lower case will be looked up.


        Returns
        -------
        mxnet.ndarray.NDArray:
            The embedding vector(s) of the token(s). According to numpy
            conventions, if `tokens` is a string, returns a 1-D NDArray of shape
            `self.vec_len`; if `tokens` is a list of strings, returns a 2-D
            NDArray of shape=(len(tokens), self.vec_len).
        """

        to_reduce = False
        if not isinstance(tokens, list):
            tokens = [tokens]
            to_reduce = True

        if not lower_case_backup:
            indices = [self.token_to_idx.get(token, C.UNKNOWN_IDX)
                       for token in tokens]
        else:
            indices = [self.token_to_idx[token] if token in self.token_to_idx
                       else self.token_to_idx.get(token.lower(), C.UNKNOWN_IDX)
                       for token in tokens]

        vecs = nd.Embedding(nd.array(indices), self.idx_to_vec,
                            self.idx_to_vec.shape[0], self.idx_to_vec.shape[1])

        return vecs[0] if to_reduce else vecs

    def update_token_vectors(self, tokens, new_vectors):
        """Updates embedding vectors for tokens.


        Parameters
        ----------
        tokens : str or a list of strs.
            A token or a list of tokens whose embedding vector are to be
            updated.
        new_vectors : mxnet.ndarray.NDArray
            An NDArray to be assigned to the embedding vectors of `tokens`.
            Its length must be equal to the number of `tokens` and its width
            must be equal to the dimension of embeddings of the glossary. If
            `tokens` is a singleton, it must be 1-D or 2-D. If `tokens` is a
            list of multiple strings, it must be 2-D.
        """

        assert self.idx_to_vec is not None, \
            'The property `idx_to_vec` has not been properly set.'

        if not isinstance(tokens, list) or len(tokens) == 1:
            assert isinstance(new_vectors, nd.NDArray) and \
                len(new_vectors.shape) in [1, 2], \
                '`new_vectors` must be a 1-D or 2-D NDArray if `tokens` is a ' \
                'singleton.'
            if not isinstance(tokens, list):
                tokens = [tokens]
            if len(new_vectors.shape) == 1:
                new_vectors = new_vectors.expand_dims(0)

        else:
            assert isinstance(new_vectors, nd.NDArray) and \
                len(new_vectors.shape) == 2, \
                '`new_vectors` must be a 2-D NDArray if `tokens` is a list ' \
                'of multiple strings.'
        assert new_vectors.shape == (len(tokens), self.vec_len), \
            'The length of new_vectors must be equal to the number of tokens ' \
            'and the width of new_vectors must be equal to the dimension of ' \
            'embeddings of the glossary.'

        indices = []
        for token in tokens:
            if token in self.token_to_idx:
                indices.append(self.token_to_idx[token])
            else:
                raise ValueError('Token %s is unknown. To update the embedding '
                                 'vector for an unknown token, please specify '
                                 'it explicitly as the `unknown_token` %s in '
                                 '`tokens`. This is to avoid unintended '
                                 'updates.' %
                                 (token, self.idx_to_token[C.UNKNOWN_IDX]))

        self._idx_to_vec[nd.array(indices)] = new_vectors

    @staticmethod
    def register(embedding_cls):
        """Registers a new token embedding.

        Once an embedding is registered, we can create an instance of this
        embedding with :func:`~mxnet.text.embedding.TokenEmbedding.create`.


        Examples
        --------
        >>> @mxnet.text.embedding.TokenEmbedding.register
        ... class MyTextEmbed(mxnet.text.embedding.TokenEmbedding):
        ...     def __init__(self, pretrained_file_name='my_pretrain_file'):
        ...         pass
        >>> embed = mxnet.text.embedding.TokenEmbedding.create('MyTokenEmbed')
        >>> print(type(embed))
        <class '__main__.MyTokenEmbed'>
        """

        register_text_embedding = registry.get_register_func(
            TokenEmbedding, 'token embedding')
        return register_text_embedding(embedding_cls)

    @staticmethod
    def create(embedding_name, **kwargs):
        """Creates an instance of :func:`~mxnet.text.embedding.TokenEmbedding`.

        Creates a token embedding instance by loading embedding vectors from an
        externally hosted pre-trained token embedding file, such as those
        of GloVe and FastText. To get all the valid `embedding_name` and
        `pretrained_file_name`, use `mxnet.text.embedding.TokenEmbedding.
        get_embedding_and_pretrained_file_names()`.


        Parameters
        ----------
        embedding_name : str
            The token embedding name (case-insensitive).


        Returns
        -------
        mxnet.text.glossary.TokenEmbedding:
            A token embedding instance that loads embedding vectors from an
            externally hosted pre-trained token embedding file.
        """

        create_text_embedding = registry.get_create_func(
            TokenEmbedding, 'token embedding')
        return create_text_embedding(embedding_name, **kwargs)

    @classmethod
    def _check_pretrained_file_names(cls, pretrained_file_name):
        """Checks if a pre-trained token embedding file name is valid.


        Parameters
        ----------
        pretrained_file_name : str
            The pre-trained token embedding file.
        """

        embedding_name = cls.__name__.lower()
        if pretrained_file_name not in cls.pretrained_file_name_sha1:
            raise KeyError('Cannot find pretrained file %s for token embedding '
                           '%s. Valid pretrained files for embedding %s: %s' %
                           (pretrained_file_name, embedding_name,
                            embedding_name,
                            ', '.join(cls.pretrained_file_name_sha1.keys())))

    @staticmethod
    def get_embedding_and_pretrained_file_names(embedding_name=None):
        """Get valid token embedding names and their pre-trained file names.

        To load token embedding vectors from an externally hosted pre-trained
        token embedding file, such as those of GloVe and FastText, one should use
        `mxnet.text.embedding.TokenEmbedding.create(embedding_name,
        pretrained_file_name)`. This method returns all the valid names of
        `pretrained_file_name` for the specified `embedding_name`. If
        `embedding_name` is set to None, this method returns all the valid names
        of `embedding_name` with associated `pretrained_file_name`.


        Parameters
        ----------
        embedding_name : str or None, default None
            The pre-trained token embedding name.


        Returns
        -------
        dict or list:
            A list of all the valid pre-trained token embedding file names
            (`pretrained_file_name`) for the specified token embedding name
            (`embedding_name`). If the text embeding name is set to None,
            returns a dict mapping each valid token embedding name to a list
            of valid pre-trained files (`pretrained_file_name`). They can be
            plugged into `mxnet.text.embedding.TokenEmbedding.create(
            embedding_name, pretrained_file_name)`.
        """

        text_embedding_reg = registry.get_registry(TokenEmbedding)

        if embedding_name is not None:
            if embedding_name not in text_embedding_reg:
                raise KeyError('Cannot find `embedding_name` %s. Use '
                               '`get_embedding_and_pretrained_file_names('
                               'embedding_name=None).keys()` to get all the '
                               'valid embedding names.' % embedding_name)
            return list(text_embedding_reg[
                embedding_name].pretrained_file_name_sha1.keys())
        else:
            return {embedding_name: list(
                embedding_cls.pretrained_file_name_sha1.keys())
                    for embedding_name, embedding_cls in
                    registry.get_registry(TokenEmbedding).items()}


@TokenEmbedding.register
class GloVe(TokenEmbedding):
    """The GloVe token embedding.

    GloVe is an unsupervised learning algorithm for obtaining vector
    representations for words. Training is performed on aggregated global
    word-word co-occurrence statistics from a corpus, and the resulting
    representations showcase interesting linear substructures of the word vector
    space. (Source from https://nlp.stanford.edu/projects/glove/)

    Reference:
    GloVe: Global Vectors for Word Representation
    Jeffrey Pennington, Richard Socher, and Christopher D. Manning
    https://nlp.stanford.edu/pubs/glove.pdf

    Website:
    https://nlp.stanford.edu/projects/glove/

    To get the updated URLs to the externally hosted pre-trained token embedding
    files, visit https://nlp.stanford.edu/projects/glove/


    Parameters
    ----------
    pretrain_file : str, default 'glove.840B.300d.txt'
        The name of the pre-trained token embedding file.
    embed_root : str, default '~/.mxnet/embeddings/'
        The root directory for storing embedding-related files.
    unknown_vec : callback
        The callback used to initialize the embedding vector for the unknown
        token.
    """

    # Map a pre-trained token embedding archive file and its SHA-1 hash.
    pretrained_archive_name_sha1 = \
        {'glove.42B.300d.zip': 'f8e722b39578f776927465b71b231bae2ae8776a',
         'glove.6B.zip': 'b64e54f1877d2f735bdd000c1d7d771e25c7dfdc',
         'glove.840B.300d.zip': '8084fbacc2dee3b1fd1ca4cc534cbfff3519ed0d',
         'glove.twitter.27B.zip': 'dce69c404025a8312c323197347695e81fd529fc'}

    # Map a pre-trained token embedding file and its SHA-1 hash.
    pretrained_file_name_sha1 = \
        {'glove.42B.300d.txt': '876767977d6bd4d947c0f84d44510677bc94612a',
         'glove.6B.50d.txt': '21bf566a9d27f84d253e0cd4d4be9dcc07976a6d',
         'glove.6B.100d.txt': '16b1dbfaf35476790bd9df40c83e2dfbd05312f1',
         'glove.6B.200d.txt': '17d0355ddaa253e298ede39877d1be70f99d9148',
         'glove.6B.300d.txt': '646443dd885090927f8215ecf7a677e9f703858d',
         'glove.840B.300d.txt': '294b9f37fa64cce31f9ebb409c266fc379527708',
         'glove.twitter.27B.25d.txt':
             '767d80889d8c8a22ae7cd25e09d0650a6ff0a502',
         'glove.twitter.27B.50d.txt':
             '9585f4be97e286339bf0112d0d3aa7c15a3e864d',
         'glove.twitter.27B.100d.txt':
             '1bbeab8323c72332bd46ada0fc3c99f2faaa8ca8',
         'glove.twitter.27B.200d.txt':
             '7921c77a53aa5977b1d9ce3a7c4430cbd9d1207a'}

    url_prefix = 'http://nlp.stanford.edu/data/'

    def __init__(self, pretrained_file_name='glove.840B.300d.txt',
                 embedding_root='~/.mxnet/embeddings/',
                 init_unknown_vec=nd.zeros, **kwargs):
        GloVe._check_pretrained_file_names(pretrained_file_name)
        src_archive = {archive.split('.')[1]: archive for archive in
                       GloVe.pretrained_archive_name_sha1.keys()}
        archive = src_archive[pretrained_file_name.split('.')[1]]
        url = GloVe.url_prefix + archive

        super(GloVe, self).__init__(**kwargs)

        pretrained_file_path = GloVe._get_pretrained_file_path_from_url(
            url, embedding_root, pretrained_file_name)

        self._load_embedding(pretrained_file_path, ' ', init_unknown_vec)


@TokenEmbedding.register
class FastText(TokenEmbedding):
    """The fastText token embedding.

    FastText is an open-source, free, lightweight library that allows users to
    learn text representations and text classifiers. It works on standard,
    generic hardware. Models can later be reduced in size to even fit on mobile
    devices. (Source from https://fasttext.cc/)

    References:
    Enriching Word Vectors with Subword Information
    Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov
    https://arxiv.org/abs/1607.04606

    Bag of Tricks for Efficient Text Classification
    Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov
    https://arxiv.org/abs/1607.01759

    FastText.zip: Compressing text classification models
    Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, Herve Jegou,
    and Tomas Mikolov
    https://arxiv.org/abs/1612.03651

    Website:
    https://fasttext.cc/

    To get the updated URLs to the externally hosted pre-trained token embedding
    files, visit
    https://github.com/facebookresearch/fastText/blob/master/
    pretrained-vectors.md


    Parameters
    ----------
    pretrain_file : str, default 'wiki.en.vec'
        The name of the pre-trained token embedding file.
    embed_root : str, default '~/.mxnet/embeddings/'
        The root directory for storing embedding-related files.
    unknown_vec : callback
        The callback used to initialize the embedding vector for the unknown
        token.
    """

    # Map a pre-trained token embedding file and its SHA-1 hash.
    pretrained_file_name_sha1 = \
        {'wiki.en.vec': 'c1e418f144ceb332b4328d27addf508731fa87df',
         'wiki.simple.vec': '55267c50fbdf4e4ae0fbbda5c73830a379d68795',
         'wiki.zh.vec': '117ab34faa80e381641fbabf3a24bc8cfba44050'}
    url_prefix = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/'

    def __init__(self, pretrained_file_name='wiki.en.vec',
                 embedding_root='~/.mxnet/embeddings/',
                 init_unknown_vec=nd.zeros, **kwargs):
        FastText._check_pretrained_file_names(pretrained_file_name)
        url = FastText.url_prefix + pretrained_file_name

        super(FastText, self).__init__(**kwargs)

        pretrained_file_path = FastText._get_pretrained_file_path_from_url(
            url, embedding_root, pretrained_file_name)

        self._load_embedding(pretrained_file_path, ' ', init_unknown_vec)


class CustomEmbedding(TokenEmbedding):
    """User-defined token embedding.

    This is to load embedding vectors from a user-defined pre-trained text
    embedding file.

    Denote by v_ij the j-th element of the token embedding vector for token_i,
    the expected format of a custom pre-trained token embedding file is:

    token_1`elem_delim`v_11`elem_delim`v_12`elem_delim`...`elem_delim`v_1k\n
    token_2`elem_delim`v_21`elem_delim`v_22`elem_delim`...`elem_delim`v_2k\n
    ...

    where k is the length of the embedding vecgor `vec_len`.


    Parameters
    ----------
    pretrain_file_path : str
        The path to the custom pre-trained token embedding file.
    elem_delim : str, default ' '
        The delimiter for splitting a token and every embedding vector element
        value on the same line of the custom pre-trained token embedding file.
    unknown_vec : callback
        The callback used to initialize the embedding vector for the unknown
        token.
    """

    def __init__(self, pretrained_file_path, elem_delim=' ', encoding='utf8',
                 init_unknown_vec=nd.zeros, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self._load_embedding(pretrained_file_path, elem_delim, init_unknown_vec,
                             encoding)
