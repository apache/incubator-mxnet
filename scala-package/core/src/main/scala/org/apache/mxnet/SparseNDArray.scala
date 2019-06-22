/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mxnet

import org.apache.mxnet.Base.{NDArrayHandle, NDArrayHandleRef, checkCall, _LIB}
import org.apache.mxnet.DType.DType
import org.apache.mxnet.SparseFormat.SparseFormat

object SparseNDArray {
  def csrMatrix(data: Array[Float], indices: Array[Float],
                indptr: Array[Float], shape: Shape, ctx: Context): SparseNDArray = {
    val fmt = SparseFormat.CSR
    val dataND = NDArray.array(data, Shape(data.length), ctx)
    val indicesND = NDArray.array(indices, Shape(indices.length), ctx).asType(DType.Int64)
    val indptrND = NDArray.array(indptr, Shape(indptr.length), ctx).asType(DType.Int64)
    val dTypes = Array(indptrND.dtype, indicesND.dtype)
    val shapes = Array(indptrND.shape, indicesND.shape)
    val handle =
      newAllocHandle(fmt, shape, ctx, false, DType.Float32, dTypes, shapes)
    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(handle, dataND.handle, -1))
    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(handle, indptrND.handle, 0))
    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(handle, indicesND.handle, 1))
    new SparseNDArray(handle, false)
  }

  def rowSparseArray(data: Array[_], indices: Array[Float],
                     shape: Shape, ctx: Context): SparseNDArray = {
    val dataND = NDArray.toNDArray(data)
    val indicesND = NDArray.array(indices, Shape(indices.length), ctx).asType(DType.Int64)
    rowSparseArray(dataND, indicesND, shape, ctx)
  }

  def rowSparseArray(data: NDArray, indices: NDArray,
                     shape: Shape, ctx: Context): SparseNDArray = {
    val fmt = SparseFormat.ROW_SPARSE
    val handle = newAllocHandle(fmt, shape, ctx, false,
      DType.Float32, Array(indices.dtype), Array(indices.shape))
    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(handle, data.handle, -1))
    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(handle, indices.handle, 0))
    new SparseNDArray(handle, false)
  }

  private def newAllocHandle(stype : SparseFormat,
                             shape: Shape,
                             ctx: Context,
                             delayAlloc: Boolean,
                             dtype: DType = DType.Float32,
                             auxDTypes: Array[DType],
                             auxShapes: Array[Shape]) : NDArrayHandle = {
    val hdl = new NDArrayHandleRef
    checkCall(_LIB.mxNDArrayCreateSparseEx(
      stype.id,
      shape.toArray,
      shape.length,
      ctx.deviceTypeid,
      ctx.deviceId,
      if (delayAlloc) 1 else 0,
      dtype.id,
      auxDTypes.length,
      auxDTypes.map(_.id),
      auxShapes.map(_.length),
      auxShapes.map(_.get(0)),
      hdl)
    )
    hdl.value
  }
}

class SparseNDArray private[mxnet] (override private[mxnet] val handle: NDArrayHandle,
                                    override val writable: Boolean)
  extends NDArray(handle, writable) {

  private lazy val dense: NDArray = toDense

  override def toString: String = {
    dense.toString
  }

  def toDense: NDArray = {
      NDArray.api.cast_storage(this, SparseFormat.DEFAULT.toString).head
  }

  override def toArray: Array[Float] = {
    dense.toArray
  }

  override def at(idx: Int): NDArray = {
    dense.at(idx)
  }
}
