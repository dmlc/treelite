#include <treelite/c_api_common.h>
#include <treelite/c_api_runtime.h>
#include <treelite/c_api_error.h>
#include <treelite/predictor.h>
#include <treelite/typeinfo.h>
#include <algorithm>
#include <limits>
#include <vector>
#include "./treelite4j.h"

namespace {

// set handle
void setHandle(JNIEnv* jenv, jlongArray jhandle, void* handle) {
#ifdef __APPLE__
  auto out = static_cast<jlong>(reinterpret_cast<long>(handle));
#else
  auto out = reinterpret_cast<int64_t>(handle);
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
  jstring jresult = nullptr;
  const char* result = TreeliteGetLastError();
  if (result) {
    jresult = jenv->NewStringUTF(result);
  }
  return jresult;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreeliteDMatrixCreateFromCSRWithFloat32In
 * Signature: ([F[I[JJJ[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreeliteDMatrixCreateFromCSRWithFloat32In(
    JNIEnv* jenv, jclass jcls, jfloatArray jdata, jintArray jcol_ind, jlongArray jrow_ptr,
    jlong jnum_row, jlong jnum_col, jlongArray jout) {
  jfloat* data = jenv->GetFloatArrayElements(jdata, nullptr);
  jint* col_ind = jenv->GetIntArrayElements(jcol_ind, nullptr);
  jlong* row_ptr = jenv->GetLongArrayElements(jrow_ptr, nullptr);
  DMatrixHandle out = nullptr;
  const int ret = TreeliteDMatrixCreateFromCSR(
      static_cast<const void*>(data), "float32", reinterpret_cast<const uint32_t*>(col_ind),
      reinterpret_cast<const size_t*>(row_ptr), static_cast<size_t>(jnum_row),
      static_cast<size_t>(jnum_col), &out);
  setHandle(jenv, jout, out);
  // release arrays
  jenv->ReleaseFloatArrayElements(jdata, data, 0);
  jenv->ReleaseIntArrayElements(jcol_ind, col_ind, 0);
  jenv->ReleaseLongArrayElements(jrow_ptr, row_ptr, 0);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreeliteDMatrixCreateFromCSRWithFloat64In
 * Signature: ([D[I[JJJ[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreeliteDMatrixCreateFromCSRWithFloat64In(
    JNIEnv* jenv, jclass jcls, jdoubleArray jdata, jintArray jcol_ind, jlongArray jrow_ptr,
    jlong jnum_row, jlong jnum_col, jlongArray jout) {
  jdouble* data = jenv->GetDoubleArrayElements(jdata, nullptr);
  jint* col_ind = jenv->GetIntArrayElements(jcol_ind, nullptr);
  jlong* row_ptr = jenv->GetLongArrayElements(jrow_ptr, nullptr);
  DMatrixHandle out = nullptr;
  const int ret = TreeliteDMatrixCreateFromCSR(
      static_cast<const void*>(data), "float64", reinterpret_cast<const uint32_t*>(col_ind),
      reinterpret_cast<const size_t*>(row_ptr), static_cast<size_t>(jnum_row),
      static_cast<size_t>(jnum_col), &out);
  setHandle(jenv, jout, out);
  // release arrays
  jenv->ReleaseDoubleArrayElements(jdata, data, 0);
  jenv->ReleaseIntArrayElements(jcol_ind, col_ind, 0);
  jenv->ReleaseLongArrayElements(jrow_ptr, row_ptr, 0);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreeliteDMatrixCreateFromMatWithFloat32In
 * Signature: ([FJJF[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreeliteDMatrixCreateFromMatWithFloat32In(
    JNIEnv* jenv, jclass jcls, jfloatArray jdata, jlong jnum_row, jlong jnum_col,
    jfloat jmissing_value, jlongArray jout) {
  jfloat* data = jenv->GetFloatArrayElements(jdata, nullptr);
  float missing_value = static_cast<float>(jmissing_value);
  DMatrixHandle out = nullptr;
  const int ret = TreeliteDMatrixCreateFromMat(
      static_cast<const void*>(data), "float32", static_cast<size_t>(jnum_row),
      static_cast<size_t>(jnum_col), &missing_value, &out);
  setHandle(jenv, jout, out);
  // release arrays
  jenv->ReleaseFloatArrayElements(jdata, data, 0);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreeliteDMatrixCreateFromMatWithFloat64In
 * Signature: ([DJJD[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreeliteDMatrixCreateFromMatWithFloat64In(
    JNIEnv* jenv, jclass jcls, jdoubleArray jdata, jlong jnum_row, jlong jnum_col,
    jdouble jmissing_value, jlongArray jout) {
  jdouble* data = jenv->GetDoubleArrayElements(jdata, nullptr);
  double missing_value = static_cast<double>(jmissing_value);
  DMatrixHandle out = nullptr;
  const int ret = TreeliteDMatrixCreateFromMat(
      static_cast<const void*>(data), "float64", static_cast<size_t>(jnum_row),
      static_cast<size_t>(jnum_col), &missing_value, &out);
  setHandle(jenv, jout, out);
  // release arrays
  jenv->ReleaseDoubleArrayElements(jdata, data, 0);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreeliteDMatrixGetDimension
 * Signature: (J[J[J[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreeliteDMatrixGetDimension(
    JNIEnv* jenv, jclass jcls, jlong jmat, jlongArray jout_num_row, jlongArray jout_num_col,
    jlongArray jout_nelem) {
  DMatrixHandle dmat = reinterpret_cast<DMatrixHandle>(jmat);
  size_t num_row = 0, num_col = 0, num_elem = 0;
  const int ret = TreeliteDMatrixGetDimension(dmat, &num_row, &num_col, &num_elem);
  // save dimensions
  jlong* out_num_row = jenv->GetLongArrayElements(jout_num_row, nullptr);
  jlong* out_num_col = jenv->GetLongArrayElements(jout_num_col, nullptr);
  jlong* out_nelem = jenv->GetLongArrayElements(jout_nelem, nullptr);
  out_num_row[0] = static_cast<jlong>(num_row);
  out_num_col[0] = static_cast<jlong>(num_col);
  out_nelem[0] = static_cast<jlong>(num_elem);
  // release arrays
  jenv->ReleaseLongArrayElements(jout_num_row, out_num_row, 0);
  jenv->ReleaseLongArrayElements(jout_num_col, out_num_col, 0);
  jenv->ReleaseLongArrayElements(jout_nelem, out_nelem, 0);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreeliteDMatrixFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreeliteDMatrixFree(
    JNIEnv* jenv, jclass jcls, jlong jdmat) {
  return static_cast<int>(TreeliteDMatrixFree(reinterpret_cast<DMatrixHandle>(jdmat)));
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorLoad
 * Signature: (Ljava/lang/String;I[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorLoad(
    JNIEnv* jenv, jclass jcls, jstring jlibrary_path, jint jnum_worker_thread, jlongArray jout) {
  const char* library_path = jenv->GetStringUTFChars(jlibrary_path, nullptr);
  PredictorHandle out = nullptr;
  const int ret = TreelitePredictorLoad(library_path, static_cast<int>(jnum_worker_thread), &out);
  setHandle(jenv, jout, out);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorPredictBatchWithFloat32Out
 * Signature: (JJZZ[F[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorPredictBatchWithFloat32Out(
    JNIEnv* jenv, jclass jcls, jlong jpredictor, jlong jbatch, jboolean jverbose,
    jboolean jpred_margin, jfloatArray jout_result, jlongArray jout_result_size) {
  PredictorHandle predictor = reinterpret_cast<PredictorHandle>(jpredictor);
  DMatrixHandle dmat = reinterpret_cast<DMatrixHandle>(jbatch);
  jfloat* out_result = jenv->GetFloatArrayElements(jout_result, nullptr);
  jlong* out_result_size = jenv->GetLongArrayElements(jout_result_size, nullptr);
  size_t out_result_size_tmp = 0;
  const int ret = TreelitePredictorPredictBatch(
      predictor, dmat, (jverbose == JNI_TRUE ? 1 : 0),
      (jpred_margin == JNI_TRUE ? 1 : 0), static_cast<void*>(out_result),
      &out_result_size_tmp);
  out_result_size[0] = static_cast<jlong>(out_result_size_tmp);
  // release arrays
  jenv->ReleaseFloatArrayElements(jout_result, out_result, 0);
  jenv->ReleaseLongArrayElements(jout_result_size, out_result_size, 0);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorPredictBatchWithFloat64Out
 * Signature: (JJZZ[D[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorPredictBatchWithFloat64Out(
    JNIEnv* jenv, jclass jcls, jlong jpredictor, jlong jbatch, jboolean jverbose,
    jboolean jpred_margin, jdoubleArray jout_result, jlongArray jout_result_size) {
  PredictorHandle predictor = reinterpret_cast<PredictorHandle>(jpredictor);
  DMatrixHandle dmat = reinterpret_cast<DMatrixHandle>(jbatch);
  jdouble* out_result = jenv->GetDoubleArrayElements(jout_result, nullptr);
  jlong* out_result_size = jenv->GetLongArrayElements(jout_result_size, nullptr);
  size_t out_result_size_tmp = 0;
  const int ret = TreelitePredictorPredictBatch(
      predictor, dmat, (jverbose == JNI_TRUE ? 1 : 0),
      (jpred_margin == JNI_TRUE ? 1 : 0), static_cast<void*>(out_result),
      &out_result_size_tmp);
  out_result_size[0] = static_cast<jlong>(out_result_size_tmp);
  // release arrays
  jenv->ReleaseDoubleArrayElements(jout_result, out_result, 0);
  jenv->ReleaseLongArrayElements(jout_result_size, out_result_size, 0);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorPredictBatchWithUInt32Out
 * Signature: (JJZZ[I[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorPredictBatchWithUInt32Out(
    JNIEnv* jenv, jclass jcls, jlong jpredictor, jlong jbatch, jboolean jverbose,
    jboolean jpred_margin, jintArray jout_result, jlongArray jout_result_size) {
  API_BEGIN();
  PredictorHandle predictor = reinterpret_cast<PredictorHandle>(jpredictor);
  DMatrixHandle dmat = reinterpret_cast<DMatrixHandle>(jbatch);
  TREELITE_CHECK_EQ(sizeof(jint), sizeof(uint32_t));
  jint* out_result = jenv->GetIntArrayElements(jout_result, nullptr);
  jlong* out_result_size = jenv->GetLongArrayElements(jout_result_size, nullptr);
  size_t out_result_size_tmp = 0;
  const int ret = TreelitePredictorPredictBatch(
      predictor, dmat, (jverbose == JNI_TRUE ? 1 : 0),
      (jpred_margin == JNI_TRUE ? 1 : 0), static_cast<void*>(out_result),
      &out_result_size_tmp);
  out_result_size[0] = static_cast<jlong>(out_result_size_tmp);
  // release arrays
  jenv->ReleaseIntArrayElements(jout_result, out_result, 0);
  jenv->ReleaseLongArrayElements(jout_result_size, out_result_size, 0);

  return static_cast<jint>(ret);
  API_END();
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQueryResultSize
 * Signature: (JJ[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQueryResultSize(
    JNIEnv* jenv, jclass jcls, jlong jpredictor, jlong jbatch, jlongArray jout) {
  PredictorHandle predictor = reinterpret_cast<PredictorHandle>(jpredictor);
  DMatrixHandle dmat = reinterpret_cast<DMatrixHandle>(jbatch);
  size_t result_size = 0;
  const int ret = TreelitePredictorQueryResultSize(predictor, dmat, &result_size);
  // store dimension
  jlong* out = jenv->GetLongArrayElements(jout, nullptr);
  out[0] = static_cast<jlong>(result_size);
  jenv->ReleaseLongArrayElements(jout, out, 0);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQueryNumClass
 * Signature: (J[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQueryNumClass(
    JNIEnv* jenv, jclass jcls, jlong jpredictor, jlongArray jout) {
  PredictorHandle predictor = reinterpret_cast<PredictorHandle>(jpredictor);
  size_t num_class = 0;
  const int ret = TreelitePredictorQueryNumClass(predictor, &num_class);
  // store dimension
  jlong* out = jenv->GetLongArrayElements(jout, nullptr);
  out[0] = static_cast<jlong>(num_class);
  jenv->ReleaseLongArrayElements(jout, out, 0);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQueryNumFeature
 * Signature: (J[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQueryNumFeature(
    JNIEnv* jenv, jclass jcls, jlong jpredictor, jlongArray jout) {
  PredictorHandle predictor = reinterpret_cast<PredictorHandle>(jpredictor);
  size_t num_feature = 0;
  const int ret = TreelitePredictorQueryNumFeature(predictor, &num_feature);
  // store dimension
  jlong* out = jenv->GetLongArrayElements(jout, nullptr);
  out[0] = static_cast<jlong>(num_feature);
  jenv->ReleaseLongArrayElements(jout, out, 0);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQueryPredTransform
 * Signature: (J[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQueryPredTransform(
    JNIEnv* jenv, jclass jcls, jlong jpredictor, jobjectArray jout) {
  PredictorHandle predictor = reinterpret_cast<PredictorHandle>(jpredictor);
  const char* pred_transform = nullptr;
  const int ret = TreelitePredictorQueryPredTransform(predictor, &pred_transform);
  // store data
  jstring out = nullptr;
  if (pred_transform != nullptr) {
    out = jenv->NewStringUTF(pred_transform);
  }
  jenv->SetObjectArrayElement(jout, 0, out);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQuerySigmoidAlpha
 * Signature: (J[F)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQuerySigmoidAlpha(
    JNIEnv* jenv, jclass jcls, jlong jpredictor, jfloatArray jout) {
  PredictorHandle predictor = reinterpret_cast<PredictorHandle>(jpredictor);
  float alpha = std::numeric_limits<float>::quiet_NaN();
  const int ret = TreelitePredictorQuerySigmoidAlpha(predictor, &alpha);
  // store data
  jfloat* out = jenv->GetFloatArrayElements(jout, nullptr);
  out[0] = static_cast<jfloat>(alpha);
  jenv->ReleaseFloatArrayElements(jout, out, 0);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQueryGlobalBias
 * Signature: (J[F)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQueryGlobalBias(
    JNIEnv* jenv, jclass jcls, jlong jpredictor, jfloatArray jout) {
  PredictorHandle predictor = reinterpret_cast<PredictorHandle>(jpredictor);
  float bias = std::numeric_limits<float>::quiet_NaN();
  const int ret = TreelitePredictorQueryGlobalBias(predictor, &bias);
  // store data
  jfloat* out = jenv->GetFloatArrayElements(jout, nullptr);
  out[0] = static_cast<jfloat>(bias);
  jenv->ReleaseFloatArrayElements(jout, out, 0);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQueryThresholdType
 * Signature: (J[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQueryThresholdType(
    JNIEnv* jenv, jclass jcls, jlong jpredictor, jobjectArray jout) {
  PredictorHandle predictor = reinterpret_cast<PredictorHandle>(jpredictor);
  const char* threshold_type = nullptr;
  const int ret = TreelitePredictorQueryThresholdType(predictor, &threshold_type);
  // store data
  jstring out = nullptr;
  if (threshold_type != nullptr) {
    out = jenv->NewStringUTF(threshold_type);
  }
  jenv->SetObjectArrayElement(jout, 0, out);

  return static_cast<jint>(ret);
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorQueryLeafOutputType
 * Signature: (J[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorQueryLeafOutputType(
    JNIEnv* jenv, jclass jcls, jlong jpredictor, jobjectArray jout) {
  PredictorHandle predictor = reinterpret_cast<PredictorHandle>(jpredictor);
  const char* leaf_output_type = nullptr;
  const jint ret = (jint)TreelitePredictorQueryLeafOutputType(predictor, &leaf_output_type);
  // store data
  jstring out = nullptr;
  if (leaf_output_type != nullptr) {
    out = jenv->NewStringUTF(leaf_output_type);
  }
  jenv->SetObjectArrayElement(jout, 0, out);

  return ret;
}

/*
 * Class:     ml_dmlc_treelite4j_java_TreeliteJNI
 * Method:    TreelitePredictorFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_java_TreeliteJNI_TreelitePredictorFree(
    JNIEnv* jenv, jclass jcls, jlong jpredictor) {
  PredictorHandle predictor = reinterpret_cast<PredictorHandle>(jpredictor);
  return static_cast<jint>(TreelitePredictorFree(predictor));
}
