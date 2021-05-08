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

"""Read and write for the RecordIO data format."""
from collections import namedtuple
from multiprocessing import current_process
from threading import get_ident

import ctypes
import struct
import numbers
import numpy as np

from .base import _LIB
from .base import RecordIOHandle
from .base import check_call
from .base import c_str
try:
    import cv2
except ImportError:
    cv2 = None

class MXRecordIO(object):
    """Reads/writes `RecordIO` data format, supporting sequential read and write.

    Examples
    ---------
    >>> record = mx.recordio.MXRecordIO('tmp.rec', 'w')
    <mxnet.recordio.MXRecordIO object at 0x10ef40ed0>
    >>> for i in range(5):
    ...    record.write('record_%d'%i)
    >>> record.close()
    >>> record = mx.recordio.MXRecordIO('tmp.rec', 'r')
    >>> for i in range(5):
    ...    item = record.read()
    ...    print(item)
    record_0
    record_1
    record_2
    record_3
    record_4
    >>> record.close()

    Parameters
    ----------
    uri : string
        Path to the record file.
    flag : string
        'w' for write or 'r' for read.
    """
    def __init__(self, uri, flag):
        self.uri = c_str(uri)
        self.handle = RecordIOHandle()
        self.flag = flag
        self.pid = None
        self.tid = None
        self.is_open = False
        self.open()

    def open(self):
        """Opens the record file."""
        if self.flag == "w":
            check_call(_LIB.MXRecordIOWriterCreate(self.uri, ctypes.byref(self.handle)))
            self.writable = True
        elif self.flag == "r":
            check_call(_LIB.MXRecordIOReaderCreate(self.uri, ctypes.byref(self.handle)))
            self.writable = False
        else:
            raise ValueError("Invalid flag %s"%self.flag)
        # pylint: disable=not-callable
        # It's bug from pylint(astroid). See https://github.com/PyCQA/pylint/issues/1699
        self.pid = current_process().pid
        self.tid = get_ident()
        self.is_open = True

    def __del__(self):
        self.close()

    def __getstate__(self):
        """Override pickling behavior."""
        # pickling pointer is not allowed
        is_open = self.is_open
        self.close()
        d = dict(self.__dict__)
        d['is_open'] = is_open
        uri = self.uri.value
        try:
            uri = uri.decode('utf-8')
        except AttributeError:
            pass
        del d['handle']
        d['uri'] = uri
        return d

    def __setstate__(self, d):
        """Restore from pickled."""
        self.__dict__ = d
        is_open = d['is_open']
        self.is_open = False
        self.handle = RecordIOHandle()
        self.uri = c_str(self.uri)
        if is_open:
            self.open()

    def _check_pid(self, allow_reset=False):
        """Check process id to ensure integrity, reset if in new process."""
        # pylint: disable=not-callable
        # It's bug from pylint(astroid). See https://github.com/PyCQA/pylint/issues/1699
        if not self.pid == current_process().pid or not self.tid == get_ident():
            if allow_reset:
                self.reset()
            else:
                raise RuntimeError("Forbidden operation in multiple processes")

    def close(self):
        """Closes the record file."""
        if not self.is_open:
            return
        if self.writable:
            check_call(_LIB.MXRecordIOWriterFree(self.handle))
        else:
            check_call(_LIB.MXRecordIOReaderFree(self.handle))
        self.is_open = False
        self.pid = None
        self.tid = None

    def reset(self):
        """Resets the pointer to first item.

        If the record is opened with 'w', this function will truncate the file to empty.

        Examples
        ---------
        >>> record = mx.recordio.MXRecordIO('tmp.rec', 'r')
        >>> for i in range(2):
        ...    item = record.read()
        ...    print(item)
        record_0
        record_1
        >>> record.reset()  # Pointer is reset.
        >>> print(record.read()) # Started reading from start again.
        record_0
        >>> record.close()
        """
        self.close()
        self.open()

    def write(self, buf):
        """Inserts a string buffer as a record.

        Examples
        ---------
        >>> record = mx.recordio.MXRecordIO('tmp.rec', 'w')
        >>> for i in range(5):
        ...    record.write('record_%d'%i)
        >>> record.close()

        Parameters
        ----------
        buf : string (python2), bytes (python3)
            Buffer to write.
        """
        assert self.writable
        self._check_pid(allow_reset=False)
        check_call(_LIB.MXRecordIOWriterWriteRecord(self.handle,
                                                    ctypes.c_char_p(buf),
                                                    ctypes.c_size_t(len(buf))))

    def read(self):
        """Returns record as a string.

        Examples
        ---------
        >>> record = mx.recordio.MXRecordIO('tmp.rec', 'r')
        >>> for i in range(5):
        ...    item = record.read()
        ...    print(item)
        record_0
        record_1
        record_2
        record_3
        record_4
        >>> record.close()

        Returns
        ----------
        buf : string
            Buffer read.
        """
        assert not self.writable
        # trying to implicitly read from multiple processes is forbidden,
        # there's no elegant way to handle unless lock is introduced
        self._check_pid(allow_reset=False)
        buf = ctypes.c_char_p()
        size = ctypes.c_size_t()
        check_call(_LIB.MXRecordIOReaderReadRecord(self.handle,
                                                   ctypes.byref(buf),
                                                   ctypes.byref(size)))
        if buf:
            buf = ctypes.cast(buf, ctypes.POINTER(ctypes.c_char*size.value))
            return buf.contents.raw
        else:
            return None

class MXIndexedRecordIO(MXRecordIO):
    """Reads/writes `RecordIO` data format, supporting random access.

    Examples
    ---------
    >>> for i in range(5):
    ...     record.write_idx(i, 'record_%d'%i)
    >>> record.close()
    >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
    >>> record.read_idx(3)
    record_3

    Parameters
    ----------
    idx_path : str
        Path to the index file.
    uri : str
        Path to the record file. Only supports seekable file types.
    flag : str
        'w' for write or 'r' for read.
    key_type : type
        Data type for keys.
    """
    def __init__(self, idx_path, uri, flag, key_type=int):
        self.idx_path = idx_path
        self.idx = {}
        self.keys = []
        self.key_type = key_type
        self.fidx = None
        super(MXIndexedRecordIO, self).__init__(uri, flag)

    def open(self):
        super(MXIndexedRecordIO, self).open()
        self.idx = {}
        self.keys = []
        self.fidx = open(self.idx_path, self.flag)
        if not self.writable:
            for line in iter(self.fidx.readline, ''):
                line = line.strip().split('\t')
                key = self.key_type(line[0])
                self.idx[key] = int(line[1])
                self.keys.append(key)

    def close(self):
        """Closes the record file."""
        if not self.is_open:
            return
        super(MXIndexedRecordIO, self).close()
        self.fidx.close()

    def __getstate__(self):
        """Override pickling behavior."""
        d = super(MXIndexedRecordIO, self).__getstate__()
        d['fidx'] = None
        return d

    def seek(self, idx):
        """Sets the current read pointer position.

        This function is internally called by `read_idx(idx)` to find the current
        reader pointer position. It doesn't return anything."""
        assert not self.writable
        self._check_pid(allow_reset=True)
        pos = ctypes.c_size_t(self.idx[idx])
        check_call(_LIB.MXRecordIOReaderSeek(self.handle, pos))

    def tell(self):
        """Returns the current position of write head.

        Examples
        ---------
        >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
        >>> print(record.tell())
        0
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        ...     print(record.tell())
        16
        32
        48
        64
        80
        """
        assert self.writable
        pos = ctypes.c_size_t()
        check_call(_LIB.MXRecordIOWriterTell(self.handle, ctypes.byref(pos)))
        return pos.value

    def read_idx(self, idx):
        """Returns the record at given index.

        Examples
        ---------
        >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        >>> record.close()
        >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
        >>> record.read_idx(3)
        record_3
        """
        self.seek(idx)
        return self.read()

    def write_idx(self, idx, buf):
        """Inserts input record at given index.

        Examples
        ---------
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        >>> record.close()

        Parameters
        ----------
        idx : int
            Index of a file.
        buf :
            Record to write.
        """
        key = self.key_type(idx)
        pos = self.tell()
        self.write(buf)
        self.fidx.write('%s\t%d\n'%(str(key), pos))
        self.idx[key] = pos
        self.keys.append(key)


IRHeader = namedtuple('HEADER', ['flag', 'label', 'id', 'id2'])
"""An alias for HEADER. Used to store metadata (e.g. labels) accompanying a record.
See mxnet.recordio.pack and mxnet.recordio.pack_img for example uses.

Parameters
----------
    flag : int
        Available for convenience, can be set arbitrarily.
    label : float or an array of float
        Typically used to store label(s) for a record.
    id: int
        Usually a unique id representing record.
    id2: int
        Higher order bits of the unique id, should be set to 0 (in most cases).
"""
_IR_FORMAT = 'IfQQ'
_IR_SIZE = struct.calcsize(_IR_FORMAT)

def pack(header, s):
    """Pack a string into MXImageRecord.

    Parameters
    ----------
    header : IRHeader
        Header of the image record.
        ``header.label`` can be a number or an array. See more detail in ``IRHeader``.
    s : str
        Raw image string to be packed.

    Returns
    -------
    s : str
        The packed string.

    Examples
    --------
    >>> label = 4 # label can also be a 1-D array, for example: label = [1,2,3]
    >>> id = 2574
    >>> header = mx.recordio.IRHeader(0, label, id, 0)
    >>> with open(path, 'r') as file:
    ...     s = file.read()
    >>> packed_s = mx.recordio.pack(header, s)
    """
    header = IRHeader(*header)
    if isinstance(header.label, numbers.Number):
        header = header._replace(flag=0)
    else:
        label = np.asarray(header.label, dtype=np.float32)
        header = header._replace(flag=label.size, label=0)
        s = label.tostring() + s
    s = struct.pack(_IR_FORMAT, *header) + s
    return s

def unpack(s):
    """Unpack a MXImageRecord to string.

    Parameters
    ----------
    s : str
        String buffer from ``MXRecordIO.read``.

    Returns
    -------
    header : IRHeader
        Header of the image record.
    s : str
        Unpacked string.

    Examples
    --------
    >>> record = mx.recordio.MXRecordIO('test.rec', 'r')
    >>> item = record.read()
    >>> header, s = mx.recordio.unpack(item)
    >>> header
    HEADER(flag=0, label=14.0, id=20129312, id2=0)
    """
    header = IRHeader(*struct.unpack(_IR_FORMAT, s[:_IR_SIZE]))
    s = s[_IR_SIZE:]
    if header.flag > 0:
        header = header._replace(label=np.frombuffer(s, np.float32, header.flag))
        s = s[header.flag*4:]
    return header, s

def unpack_img(s, iscolor=-1):
    """Unpack a MXImageRecord to image.

    Parameters
    ----------
    s : str
        String buffer from ``MXRecordIO.read``.
    iscolor : int
        Image format option for ``cv2.imdecode``.

    Returns
    -------
    header : IRHeader
        Header of the image record.
    img : numpy.ndarray
        Unpacked image.

    Examples
    --------
    >>> record = mx.recordio.MXRecordIO('test.rec', 'r')
    >>> item = record.read()
    >>> header, img = mx.recordio.unpack_img(item)
    >>> header
    HEADER(flag=0, label=14.0, id=20129312, id2=0)
    >>> img
    array([[[ 23,  27,  45],
            [ 28,  32,  50],
            ...,
            [ 36,  40,  59],
            [ 35,  39,  58]],
           ...,
           [[ 91,  92, 113],
            [ 97,  98, 119],
            ...,
            [168, 169, 167],
            [166, 167, 165]]], dtype=uint8)
    """
    header, s = unpack(s)
    img = np.frombuffer(s, dtype=np.uint8)
    assert cv2 is not None
    img = cv2.imdecode(img, iscolor)
    return header, img

def pack_img(header, img, quality=95, img_fmt='.jpg'):
    """Pack an image into ``MXImageRecord``.

    Parameters
    ----------
    header : IRHeader
        Header of the image record.
        ``header.label`` can be a number or an array. See more detail in ``IRHeader``.
    img : numpy.ndarray
        Image to be packed.
    quality : int
        Quality for JPEG encoding in range 1-100, or compression for PNG encoding in range 1-9.
    img_fmt : str
        Encoding of the image (.jpg for JPEG, .png for PNG).

    Returns
    -------
    s : str
        The packed string.

    Examples
    --------
    >>> label = 4 # label can also be a 1-D array, for example: label = [1,2,3]
    >>> id = 2574
    >>> header = mx.recordio.IRHeader(0, label, id, 0)
    >>> img = cv2.imread('test.jpg')
    >>> packed_s = mx.recordio.pack_img(header, img)
    """
    assert cv2 is not None
    jpg_formats = ['.JPG', '.JPEG']
    png_formats = ['.PNG']
    encode_params = None
    if img_fmt.upper() in jpg_formats:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif img_fmt.upper() in png_formats:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, quality]

    ret, buf = cv2.imencode(img_fmt, img, encode_params)
    assert ret, 'failed to encode image'
    return pack(header, buf.tostring())
