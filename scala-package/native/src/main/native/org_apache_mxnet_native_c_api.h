/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_apache_mxnet_LibInfo */

#ifndef _Included_org_apache_mxnet_LibInfo
#define _Included_org_apache_mxnet_LibInfo
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    nativeLibInit
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_nativeLibInit
  (JNIEnv *, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxGetLastError
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_org_apache_mxnet_LibInfo_mxGetLastError
  (JNIEnv *, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxListAllOpNames
 * Signature: (Lscala/collection/mutable/ListBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxListAllOpNames
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    nnGetOpHandle
 * Signature: (Ljava/lang/String;Lorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_nnGetOpHandle
  (JNIEnv *, jobject, jstring, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxImperativeInvokeEx
 * Signature: (J[J[JLscala/collection/mutable/ArrayBuffer;I[Ljava/lang/String;[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxImperativeInvokeEx
  (JNIEnv *, jobject, jlong, jlongArray, jlongArray, jobject,
   jint, jobjectArray, jobjectArray, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayFree
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayCreateNone
 * Signature: (Lorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayCreateNone
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayCreateEx
 * Signature: ([Lorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayCreateEx
  (JNIEnv *, jobject, jintArray, jint, jint, jint, jint, jint, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayCreateSparseEx
 * Signature: ([Lorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayCreateSparseEx
  (JNIEnv *, jobject, jint, jintArray, jint, jint, jint, jint, jint, jint, jintArray, jintArray, jintArray, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayWaitAll
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayWaitAll
  (JNIEnv *, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayWaitToRead
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayWaitToRead
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxListFunctions
 * Signature: (Lscala/collection/mutable/ListBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxListFunctions
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxFuncDescribe
 * Signature: (JLorg/apache/mxnet/Base/RefInt;Lorg/apache/mxnet/Base/RefInt;Lorg/apache/mxnet/Base/RefInt;Lorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxFuncDescribe
  (JNIEnv *, jobject, jlong, jobject, jobject, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxFuncGetInfo
 * Signature: (JLorg/apache/mxnet/Base/RefString;Lorg/apache/mxnet/Base/RefString;Lorg/apache/mxnet/Base/RefInt;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxFuncGetInfo
  (JNIEnv *, jobject, jlong, jobject, jobject, jobject, jobject, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxFuncInvoke
 * Signature: (J[J[F[J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxFuncInvoke
  (JNIEnv *, jobject, jlong, jlongArray, jfloatArray, jlongArray);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxFuncInvokeEx
 * Signature: (J[J[F[JI[[B[[B)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxFuncInvokeEx
  (JNIEnv *, jobject, jlong, jlongArray, jfloatArray, jlongArray, jint, jobjectArray, jobjectArray);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayGetShape
 * Signature: (JLorg/apache/mxnet/Base/RefInt;Lscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayGetShape
  (JNIEnv *, jobject, jlong, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArraySyncCopyFromNDArray
 * Signature: (J[BI)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArraySyncCopyFromNDArray
  (JNIEnv *, jobject, jlong, jlong, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArraySyncCopyToCPU
 * Signature: (J[BI)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArraySyncCopyToCPU
  (JNIEnv *, jobject, jlong, jbyteArray, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArraySlice
 * Signature: (JIILorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArraySlice
  (JNIEnv *, jobject, jlong, jint, jint, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayAt
 * Signature: (JILorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayAt
  (JNIEnv *, jobject, jlong, jint, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayReshape64
 * Signature: (JI[JZLorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayReshape64
  (JNIEnv *, jobject, jlong, jint, jlongArray, jboolean, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArraySyncCopyFromCPU
 * Signature: (J[FI)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArraySyncCopyFromCPU
  (JNIEnv *, jobject, jlong, jfloatArray, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxFloat64NDArraySyncCopyFromCPU
 * Signature: (J[DI)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxFloat64NDArraySyncCopyFromCPU
  (JNIEnv *, jobject, jlong, jdoubleArray, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayLoad
 * Signature: (Ljava/lang/String;Lorg/apache/mxnet/Base/RefInt;Lscala/collection/mutable/ArrayBuffer;Lorg/apache/mxnet/Base/RefInt;Lscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayLoad
  (JNIEnv *, jobject, jstring, jobject, jobject, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArraySave
 * Signature: (Ljava/lang/String;[J[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArraySave
  (JNIEnv *, jobject, jstring, jlongArray, jobjectArray);


/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayGetAuxNDArray
 * Signature: (JLorg/apache/mxnet/Base/RefInt;Lorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayGetAuxNDArray
  (JNIEnv *, jobject, jlong, jint, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayGetContext
 * Signature: (JLorg/apache/mxnet/Base/RefInt;Lorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayGetContext
  (JNIEnv *, jobject, jlong, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArraySaveRawBytes
 * Signature: (JLscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArraySaveRawBytes
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayLoadFromRawBytes
 * Signature: ([BLorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayLoadFromRawBytes
  (JNIEnv *, jobject, jbyteArray, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayGetDType
 * Signature: (JLorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayGetDType
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNDArrayGetStorageType
 * Signature: (JLorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNDArrayGetStorageType
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxInitPSEnv
 * Signature: ([Ljava/lang/String;[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxInitPSEnv
  (JNIEnv *, jobject, jobjectArray, jobjectArray);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStoreRunServer
 * Signature: (JLorg/apache/mxnet/KVServerControllerCallback;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStoreRunServer
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStoreGetNumDeadNode
 * Signature: (JILorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStoreGetNumDeadNode
  (JNIEnv *, jobject, jlong, jint, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStoreCreate
 * Signature: (Ljava/lang/String;Lorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStoreCreate
  (JNIEnv *, jobject, jstring, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStoreInit
 * Signature: (JI[I[J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStoreInit
  (JNIEnv *, jobject, jlong, jint, jintArray, jlongArray);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStoreInitEx
 * Signature: (JI[Ljava/lang/String;[J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStoreInitEx
  (JNIEnv *, jobject, jlong, jint, jobjectArray, jlongArray);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStorePush
 * Signature: (JI[I[JI)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStorePush
  (JNIEnv *, jobject, jlong, jint, jintArray, jlongArray, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStorePushEx
 * Signature: (JI[Ljava/lang/String;[JI)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStorePushEx
  (JNIEnv *, jobject, jlong, jint, jobjectArray, jlongArray, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStorePull
 * Signature: (JI[I[JI)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStorePull
  (JNIEnv *, jobject, jlong, jint, jintArray, jlongArray, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStorePullEx
 * Signature: (JI[Ljava/lang/String;[JI)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStorePullEx
  (JNIEnv *, jobject, jlong, jint, jobjectArray, jlongArray, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStoreSetUpdater
 * Signature: (JLorg/apache/mxnet/MXKVStoreUpdater;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStoreSetUpdater
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStoreIsWorkerNode
 * Signature: (Lorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStoreIsWorkerNode
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStoreGetType
 * Signature: (JLorg/apache/mxnet/Base/RefString;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStoreGetType
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStoreSendCommmandToServers
 * Signature: (JILjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStoreSendCommmandToServers
  (JNIEnv *, jobject, jlong, jint, jstring);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStoreBarrier
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStoreBarrier
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStoreGetGroupSize
 * Signature: (JLorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStoreGetGroupSize
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStoreGetRank
 * Signature: (JLorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStoreGetRank
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStoreSetBarrierBeforeExit
 * Signature: (JI)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStoreSetBarrierBeforeExit
  (JNIEnv *, jobject, jlong, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxKVStoreFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxKVStoreFree
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxListDataIters
 * Signature: (Lscala/collection/mutable/ListBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxListDataIters
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxDataIterCreateIter
 * Signature: (J[Ljava/lang/String;[Ljava/lang/String;Lorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxDataIterCreateIter
  (JNIEnv *, jobject, jlong, jobjectArray, jobjectArray, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxDataIterGetIterInfo
 * Signature: (JLorg/apache/mxnet/Base/RefString;Lorg/apache/mxnet/Base/RefString;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxDataIterGetIterInfo
  (JNIEnv *, jobject, jlong, jobject, jobject, jobject, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxDataIterFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxDataIterFree
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxDataIterBeforeFirst
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxDataIterBeforeFirst
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxDataIterNext
 * Signature: (JLorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxDataIterNext
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxDataIterGetLabel
 * Signature: (JLorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxDataIterGetLabel
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxDataIterGetData
 * Signature: (JLorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxDataIterGetData
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxDataIterGetIndex
 * Signature: (JLscala/collection/mutable/ListBuffer;Lorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxDataIterGetIndex
  (JNIEnv *, jobject, jlong, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxDataIterGetPadNum
 * Signature: (JLorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxDataIterGetPadNum
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxExecutorOutputs
 * Signature: (JLscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxExecutorOutputs
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxExecutorFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxExecutorFree
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxExecutorForward
 * Signature: (JI)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxExecutorForward
  (JNIEnv *, jobject, jlong, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxExecutorBackward
 * Signature: (J[J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxExecutorBackward
  (JNIEnv *, jobject, jlong, jlongArray);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxExecutorPrint
 * Signature: (JLorg/apache/mxnet/Base/RefString;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxExecutorPrint
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxExecutorSetMonitorCallback
 * Signature: (JLorg/apache/mxnet/MXMonitorCallback;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxExecutorSetMonitorCallback
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxExecutorReshape
 * Signature: (IIII[Ljava/lang/String;[I[I[Ljava/lang/String;[I[ILscala/collection/mutable/ArrayBuffer;Lscala/collection/mutable/ArrayBuffer;Lscala/collection/mutable/ArrayBuffer;JLorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxExecutorReshape
  (JNIEnv *, jobject, jint, jint, jint, jint, jobjectArray, jintArray, jintArray, jobjectArray, jintArray, jintArray, jobject, jobject, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolListAtomicSymbolCreators
 * Signature: (Lscala/collection/mutable/ListBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolListAtomicSymbolCreators
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolGetAtomicSymbolInfo
 * Signature: (JLorg/apache/mxnet/Base/RefString;Lorg/apache/mxnet/Base/RefString;Lorg/apache/mxnet/Base/RefInt;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lorg/apache/mxnet/Base/RefString;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolGetAtomicSymbolInfo
  (JNIEnv *, jobject, jlong, jobject, jobject, jobject, jobject, jobject, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolCreateAtomicSymbol
 * Signature: (J[Ljava/lang/String;[Ljava/lang/String;Lorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolCreateAtomicSymbol
  (JNIEnv *, jobject, jlong, jobjectArray, jobjectArray, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolSetAttr
 * Signature: (JLjava/lang/String;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolSetAttr
  (JNIEnv *, jobject, jlong, jstring, jstring);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolListAttrShallow
 * Signature: (JLorg/apache/mxnet/Base/RefInt;Lscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolListAttrShallow
  (JNIEnv *, jobject, jlong, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolListAttr
 * Signature: (JLorg/apache/mxnet/Base/RefInt;Lscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolListAttr
  (JNIEnv *, jobject, jlong, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolCompose
 * Signature: (JLjava/lang/String;[Ljava/lang/String;[J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolCompose
  (JNIEnv *, jobject, jlong, jstring, jobjectArray, jlongArray);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolCreateVariable
 * Signature: (Ljava/lang/String;Lorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolCreateVariable
  (JNIEnv *, jobject, jstring, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolGetAttr
 * Signature: (JLjava/lang/String;Lorg/apache/mxnet/Base/RefString;Lorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolGetAttr
  (JNIEnv *, jobject, jlong, jstring, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolListArguments
 * Signature: (JLscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolListArguments
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolCopy
 * Signature: (JLorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolCopy
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolListAuxiliaryStates
 * Signature: (JLscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolListAuxiliaryStates
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolListOutputs
 * Signature: (JLscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolListOutputs
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolCreateGroup
 * Signature: ([JLorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolCreateGroup
  (JNIEnv *, jobject, jlongArray, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolPrint
 * Signature: (JLorg/apache/mxnet/Base/RefString;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolPrint
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolGetInternals
 * Signature: (JLorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolGetInternals
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolInferType
 * Signature: (J[Ljava/lang/String;[ILscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolInferType
  (JNIEnv *, jobject, jlong, jobjectArray, jintArray, jobject, jobject, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolInferShape
 * Signature: (JI[Ljava/lang/String;[I[ILscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolInferShape
  (JNIEnv *, jobject, jlong, jint, jobjectArray, jintArray, jintArray, jobject, jobject, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolInferShapePartial
 * Signature: (JI[Ljava/lang/String;[I[ILscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolInferShapePartial
  (JNIEnv *, jobject, jlong, jint, jobjectArray, jintArray, jintArray, jobject, jobject, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolGetOutput
 * Signature: (JILorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolGetOutput
  (JNIEnv *, jobject, jlong, jint, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolSaveToJSON
 * Signature: (JLorg/apache/mxnet/Base/RefString;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolSaveToJSON
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolCreateFromJSON
 * Signature: (Ljava/lang/String;Lorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolCreateFromJSON
  (JNIEnv *, jobject, jstring, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxExecutorBindX
 * Signature: (JIII[Ljava/lang/String;[I[II[J[J[I[JLorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxExecutorBindX
  (JNIEnv *, jobject, jlong, jint, jint, jint, jobjectArray, jintArray, jintArray, jint, jlongArray, jlongArray, jintArray, jlongArray, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxExecutorBindEX
 * Signature: (JIII[Ljava/lang/String;[I[II[J[J[I[JJLorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxExecutorBindEX
  (JNIEnv *, jobject, jlong, jint, jint, jint, jobjectArray, jintArray, jintArray, jint, jlongArray, jlongArray, jintArray, jlongArray, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolSaveToFile
 * Signature: (JLjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolSaveToFile
  (JNIEnv *, jobject, jlong, jstring);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolCreateFromFile
 * Signature: (Ljava/lang/String;Lorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolCreateFromFile
  (JNIEnv *, jobject, jstring, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSymbolFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSymbolFree
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxRandomSeed
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxRandomSeed
  (JNIEnv *, jobject, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxNotifyShutdown
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxNotifyShutdown
  (JNIEnv *, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxRecordIOWriterCreate
 * Signature: (Ljava/lang/String;Lorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxRecordIOWriterCreate
  (JNIEnv *, jobject, jstring, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxRecordIOReaderCreate
 * Signature: (Ljava/lang/String;Lorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxRecordIOReaderCreate
  (JNIEnv *, jobject, jstring, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxRecordIOWriterFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxRecordIOWriterFree
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxRecordIOReaderFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxRecordIOReaderFree
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxRecordIOWriterWriteRecord
 * Signature: (JLjava/lang/String;I)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxRecordIOWriterWriteRecord
  (JNIEnv *, jobject, jlong, jstring, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxRecordIOReaderReadRecord
 * Signature: (JLorg/apache/mxnet/Base/RefString;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxRecordIOReaderReadRecord
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxRecordIOWriterTell
 * Signature: (JLorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxRecordIOWriterTell
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxRecordIOReaderSeek
 * Signature: (JI)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxRecordIOReaderSeek
  (JNIEnv *, jobject, jlong, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxRtcCreate
 * Signature: (Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;[J[JLjava/lang/String;Lorg/apache/mxnet/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxRtcCreate
  (JNIEnv *, jobject, jstring, jobjectArray, jobjectArray, jlongArray, jlongArray, jstring, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxRtcPush
 * Signature: (J[J[JIIIIII)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxRtcPush
  (JNIEnv *, jobject, jlong, jlongArray, jlongArray, jint, jint, jint, jint, jint, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxRtcFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxRtcFree
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxCustomOpRegister
 * Signature: (Ljava/lang/String;Lorg/apache/mxnet/CustomOpProp;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxCustomOpRegister
  (JNIEnv *, jobject, jstring, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSetProfilerConfig
 * Signature: ([Ljava/lang/String;[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSetProfilerConfig
  (JNIEnv *, jobject, jobjectArray, jobjectArray);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSetProfilerState
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSetProfilerState
  (JNIEnv *, jobject, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxDumpProfile
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxDumpProfile
  (JNIEnv *, jobject, jint);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxIsNumpyShape
 * Signature: (Lorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxIsNumpyShape
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_apache_mxnet_LibInfo
 * Method:    mxSetIsNumpyShape
 * Signature: (ILorg/apache/mxnet/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_mxnet_LibInfo_mxSetIsNumpyShape
  (JNIEnv *, jobject, jint, jobject);

#ifdef __cplusplus
}
#endif
#endif
