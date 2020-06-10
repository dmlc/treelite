#include <treelite/c_api_common.h>
#include <treelite/c_api_runtime.h>
#include <treelite/predictor.h>
#include <dmlc/endian.h>
#include <dmlc/logging.h>
#include <dmlc/memory_io.h>
#include <algorithm>
#include <vector>
#include "./treelite4j.h"

namespace {

// set handle
void setHandle(JNIEnv* jenv, jlongArray jhandle, void* handle) {
#ifdef __APPLE__
  jlong out = (long)handle;
#else
  int64_t out = (int64_t)handle;
#endif
  jenv->SetLongArrayRegion(jhandle, 0, 1, &out);
}

}  // namespace anonymous

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreeliteGetLastError
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreeliteGetLastError(
  JNIEnv* jenv, jclass jcls) {
  jstring jresult = 0;
  const char* result = TreeliteGetLastError();
  if (result) {
    jresult = jenv->NewStringUTF(result);
  }
  return jresult;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreeliteAssembleSparseBatch
 * Signature: ([F[I[JJJ[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreeliteAssembleSparseBatch(
  JNIEnv* jenv, jclass jcls, jfloatArray jdata, jintArray jcol_ind,
  jlongArray jrow_ptr, jlong jnum_row, jlong jnum_col, jlongArray jout) {

  jfloat* data = jenv->GetFloatArrayElements(jdata, 0);
  jint* col_ind = jenv->GetIntArrayElements(jcol_ind, 0);
  jlong* row_ptr = jenv->GetLongArrayElements(jrow_ptr, 0);
  CSRBatchHandle out;
  jint ret;
  if (sizeof(size_t) == sizeof(uint64_t)) {
    ret = (jint)TreeliteAssembleSparseBatch((const float*)data,
      (const uint32_t*)col_ind, (const size_t*)row_ptr,
      (size_t)jnum_row, (size_t)jnum_col, &out);
  } else {
    LOG(FATAL) << "32-bit platform not supported yet";
  }
  setHandle(jenv, jout, out);

  return ret;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreeliteDeleteSparseBatch
 * Signature: (J[F[I[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreeliteDeleteSparseBatch(
  JNIEnv* jenv, jclass jcls, jlong jhandle,
  jfloatArray jdata, jintArray jcol_ind, jlongArray jrow_ptr) {

  treelite::CSRBatch* batch = (treelite::CSRBatch*)jhandle;
  jenv->ReleaseFloatArrayElements(jdata, (jfloat*)batch->data, 0);
  jenv->ReleaseIntArrayElements(jcol_ind, (jint*)batch->col_ind, 0);
  jenv->ReleaseLongArrayElements(jrow_ptr, (jlong*)batch->row_ptr, 0);
  return (jint)TreeliteDeleteSparseBatch((CSRBatchHandle)batch);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreeliteAssembleDenseBatch
 * Signature: ([FFJJ[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreeliteAssembleDenseBatch(
  JNIEnv* jenv, jclass jcls, jfloatArray jdata, jfloat jmissing_value,
  jlong jnum_row, jlong jnum_col, jlongArray jout) {

  jfloat* data = jenv->GetFloatArrayElements(jdata, 0);
  DenseBatchHandle out;
  const jint ret = (jint)TreeliteAssembleDenseBatch((const float*)data,
    (float)jmissing_value, (size_t)jnum_row, (size_t)jnum_col, &out);
  setHandle(jenv, jout, out);

  return ret;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreeliteDeleteDenseBatch
 * Signature: (J[F)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreeliteDeleteDenseBatch(
  JNIEnv* jenv, jclass jcls, jlong jhandle, jfloatArray jdata) {

  treelite::DenseBatch* batch = (treelite::DenseBatch*)jhandle;
  jenv->ReleaseFloatArrayElements(jdata, (jfloat*)batch->data, 0);
  return (jint)TreeliteDeleteDenseBatch((DenseBatchHandle)batch);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreeliteBatchGetDimension
 * Signature: (JZ[J[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreeliteBatchGetDimension(
  JNIEnv* jenv, jclass jcls, jlong jhandle, jboolean jbatch_sparse,
  jlongArray jout_num_row, jlongArray jout_num_col) {

  size_t num_row, num_col;
  const jint ret = (jint)TreeliteBatchGetDimension((void*)jhandle,
    (jbatch_sparse == JNI_TRUE ? 1 : 0), &num_row, &num_col);

  // save dimensions
  jlong* out_num_row = jenv->GetLongArrayElements(jout_num_row, 0);
  jlong* out_num_col = jenv->GetLongArrayElements(jout_num_col, 0);
  out_num_row[0] = (jlong)num_row;
  out_num_col[0] = (jlong)num_col;
  jenv->ReleaseLongArrayElements(jout_num_row, out_num_row, 0);
  jenv->ReleaseLongArrayElements(jout_num_col, out_num_col, 0);

  return ret;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorLoad
 * Signature: (Ljava/lang/String;I[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorLoad(
  JNIEnv* jenv, jclass jcls, jstring jlibrary_path, jint jnum_worker_thread,
  jlongArray jout) {

  const char* library_path = jenv->GetStringUTFChars(jlibrary_path, 0);
  PredictorHandle out;
  const jint ret = (jint)TreelitePredictorLoad(library_path,
    (int)jnum_worker_thread, &out);
  setHandle(jenv, jout, out);

  return ret;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorPredictBatch
 * Signature: (JJZZZ[F)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorPredictBatch(
  JNIEnv* jenv, jclass jcls, jlong jhandle, jlong jbatch,
  jboolean jbatch_sparse, jboolean jverbose, jboolean jpred_margin,
  jfloatArray jout_result, jlongArray jout_result_size) {

  jfloat* out_result = jenv->GetFloatArrayElements(jout_result, 0);
  jlong* out_result_size = jenv->GetLongArrayElements(jout_result_size, 0);
  size_t out_result_size_tmp;
  const jint ret = (jint)TreelitePredictorPredictBatch(
    (PredictorHandle)jhandle, (void*)jbatch,
    (jbatch_sparse == JNI_TRUE ? 1 : 0), (jverbose == JNI_TRUE ? 1 : 0),
    (jpred_margin == JNI_TRUE ? 1 : 0), (float*)out_result,
    &out_result_size_tmp);
  out_result_size[0] = (jlong)out_result_size_tmp;

  // release arrays
  jenv->ReleaseFloatArrayElements(jout_result, out_result, 0);
  jenv->ReleaseLongArrayElements(jout_result_size, out_result_size, 0);

  return ret;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorPredictInst
 * Signature: (J[BZ[F[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorPredictInst(
  JNIEnv* jenv, jclass jcls, jlong jhandle, jbyteArray jinst,
  jboolean jpred_margin, jfloatArray jout_result, jlongArray jout_result_size) {

  // read Entry[] array from bytes
  jbyte* inst_bytes = jenv->GetByteArrayElements(jinst, 0);
  const size_t nbytes = jenv->GetArrayLength(jinst);
  CHECK_EQ(nbytes % sizeof(TreelitePredictorEntry), 0);
  const size_t num_elem = nbytes / sizeof(TreelitePredictorEntry);
  if (!DMLC_LITTLE_ENDIAN) {  // re-order bytes on big-endian machines
    dmlc::ByteSwap((void*)inst_bytes, nbytes, num_elem);
  }
  dmlc::MemoryFixedSizeStream fs((void*)inst_bytes, nbytes);
  std::vector<TreelitePredictorEntry> inst(num_elem, {-1});
  for (int i = 0; i < num_elem; ++i) {
    fs.Read(&inst[i], sizeof(TreelitePredictorEntry));
  }

  jfloat* out_result = jenv->GetFloatArrayElements(jout_result, 0);
  jlong* out_result_size = jenv->GetLongArrayElements(jout_result_size, 0);
  size_t out_result_size_tmp;
  const jint ret = (jint)TreelitePredictorPredictInst((PredictorHandle)jhandle,
    inst.data(), (jpred_margin == JNI_TRUE ? 1 : 0),
    (float*)out_result, &out_result_size_tmp);
  out_result_size[0] = (jlong)out_result_size_tmp;

  // release arrays
  jenv->ReleaseByteArrayElements(jinst, inst_bytes, 0);
  jenv->ReleaseFloatArrayElements(jout_result, out_result, 0);
  jenv->ReleaseLongArrayElements(jout_result_size, out_result_size, 0);

  return ret;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQueryResultSize
 * Signature: (JJZ[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQueryResultSize(
  JNIEnv* jenv, jclass jcls, jlong jhandle, jlong jbatch,
  jboolean jbatch_sparse, jlongArray jout) {

  size_t result_size;
  const jint ret = (jint)TreelitePredictorQueryResultSize(
    (PredictorHandle)jhandle, (void*)jbatch,
    (jbatch_sparse == JNI_TRUE ? 1 : 0), &result_size);
  // store dimension
  jlong* out = jenv->GetLongArrayElements(jout, 0);
  out[0] = (jlong)result_size;
  jenv->ReleaseLongArrayElements(jout, out, 0);

  return ret;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQueryResultSizeSingleInst
 * Signature: (J[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQueryResultSizeSingleInst(
  JNIEnv* jenv, jclass jcls, jlong jhandle, jlongArray jout) {
  size_t result_size;
  const jint ret = (jint)TreelitePredictorQueryResultSizeSingleInst(
    (PredictorHandle)jhandle, &result_size);
  // store dimension
  jlong* out = jenv->GetLongArrayElements(jout, 0);
  out[0] = (jlong)result_size;
  jenv->ReleaseLongArrayElements(jout, out, 0);
  return ret;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQueryNumOutputGroup
 * Signature: (J[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQueryNumOutputGroup(
  JNIEnv* jenv, jclass jcls, jlong jhandle, jlongArray jout) {

  size_t num_output_group;
  const jint ret = (jint)TreelitePredictorQueryNumOutputGroup(
    (PredictorHandle)jhandle, &num_output_group);
  // store dimension
  jlong* out = jenv->GetLongArrayElements(jout, 0);
  out[0] = (jlong)num_output_group;
  jenv->ReleaseLongArrayElements(jout, out, 0);

  return ret;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQueryNumFeature
 * Signature: (J[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQueryNumFeature(
  JNIEnv* jenv, jclass jcls, jlong jhandle, jlongArray jout) {

  size_t num_feature;
  const jint ret = (jint)TreelitePredictorQueryNumFeature(
    (PredictorHandle)jhandle, &num_feature);
  // store dimension
  jlong* out = jenv->GetLongArrayElements(jout, 0);
  out[0] = (jlong)num_feature;
  jenv->ReleaseLongArrayElements(jout, out, 0);

  return ret;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQueryPredTransform
 * Signature: (J[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQueryPredTransform(
  JNIEnv* jenv, jclass jcls, jlong jhandle, jobjectArray jout) {

  const char* pred_transform;
  const jint ret = (jint)TreelitePredictorQueryPredTransform(
      (PredictorHandle)jhandle, &pred_transform);
  // store data
  jstring out = nullptr;
  if (pred_transform != nullptr) {
    out = jenv->NewStringUTF(pred_transform);
  }
  jenv->SetObjectArrayElement(jout, 0, out);

  return ret;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQuerySigmoidAlpha
 * Signature: (J[F)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQuerySigmoidAlpha(
    JNIEnv* jenv, jclass jcls, jlong jhandle, jfloatArray jout) {

  float alpha;
  const jint ret = (jint)TreelitePredictorQuerySigmoidAlpha(
      (PredictorHandle)jhandle, &alpha);
  // store data
  jfloat* out = jenv->GetFloatArrayElements(jout, 0);
  out[0] = (jlong)alpha;
  jenv->ReleaseFloatArrayElements(jout, out, 0);

  return ret;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQueryGlobalBias
 * Signature: (J[F)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQueryGlobalBias(
    JNIEnv* jenv, jclass jcls, jlong jhandle, jfloatArray jout) {

  float bias;
  const jint ret = (jint)TreelitePredictorQueryGlobalBias(
      (PredictorHandle)jhandle, &bias);
  // store data
  jfloat* out = jenv->GetFloatArrayElements(jout, 0);
  out[0] = (jlong)bias;
  jenv->ReleaseFloatArrayElements(jout, out, 0);

  return ret;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorFree(
  JNIEnv* jenv, jclass jcls, jlong jhandle) {
  return (jint)TreelitePredictorFree((PredictorHandle)jhandle);
}
