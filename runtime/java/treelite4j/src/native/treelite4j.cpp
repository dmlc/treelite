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
 * Class:     ml_dmlc_treelite4j_TreeliteJNI
 * Method:    TreeliteGetLastError
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_ml_dmlc_treelite4j_TreeliteJNI_TreeliteGetLastError(
  JNIEnv* jenv, jclass jcls) {
  return jenv->NewStringUTF("");
}

/*
 * Class:     ml_dmlc_treelite4j_TreeliteJNI
 * Method:    TreeliteAssembleSparseBatch
 * Signature: ([F[I[JJJ[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_TreeliteJNI_TreeliteAssembleSparseBatch(
  JNIEnv* jenv, jclass jcls, jfloatArray jdata, jintArray jcol_ind,
  jlongArray jrow_ptr, jlong jnum_row, jlong jnum_col, jlongArray jout) {
  return (jint)0;
}

/*
 * Class:     ml_dmlc_treelite4j_TreeliteJNI
 * Method:    TreeliteDeleteSparseBatch
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_TreeliteJNI_TreeliteDeleteSparseBatch(
  JNIEnv* jenv, jclass jcls, jlong jhandle) {
  return (jint)0;
}

/*
 * Class:     ml_dmlc_treelite4j_TreeliteJNI
 * Method:    TreeliteAssembleDenseBatch
 * Signature: ([FFJJ[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_TreeliteJNI_TreeliteAssembleDenseBatch(
  JNIEnv* jenv, jclass jcls, jfloatArray jdata, jfloat jmissing_value,
  jlong jnum_row, jlong jnum_col, jlongArray jout) {
  return (jint)0;
}

/*
 * Class:     ml_dmlc_treelite4j_TreeliteJNI
 * Method:    TreeliteDeleteDenseBatch
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_TreeliteJNI_TreeliteDeleteDenseBatch(
  JNIEnv* jenv, jclass jcls, jlong jhandle) {
  return (jint)0;
}

/*
 * Class:     ml_dmlc_treelite4j_TreeliteJNI
 * Method:    TreeliteBatchGetDimension
 * Signature: (JZ[J[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_TreeliteJNI_TreeliteBatchGetDimension(
  JNIEnv* jenv, jclass jcls, jlong jhandle, jboolean jbatch_sparse,
  jlongArray jout_num_row, jlongArray jout_num_col) {
  return (jint)0;
}

/*
 * Class:     ml_dmlc_treelite4j_TreeliteJNI
 * Method:    TreelitePredictorLoad
 * Signature: (Ljava/lang/String;IZ[J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_TreeliteJNI_TreelitePredictorLoad(
  JNIEnv* jenv, jclass jcls, jstring jlibrary_path, jint jnum_worker_thread,
  jboolean jinclude_master_thread, jlongArray jout) {
  return (jint)0;
}

/*
 * Class:     ml_dmlc_treelite4j_TreeliteJNI
 * Method:    TreelitePredictorPredictBatch
 * Signature: (JJZZZ[F)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_TreeliteJNI_TreelitePredictorPredictBatch(
  JNIEnv* jenv, jclass jcls, jlong jhandle, jlong jbatch,
  jboolean jbatch_sparse, jboolean jverbose, jboolean jpred_margin,
  jfloatArray jout_result) {
  return (jint)0;
}

/*
 * Class:     ml_dmlc_treelite4j_TreeliteJNI
 * Method:    TreelitePredictorFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_treelite4j_TreeliteJNI_TreelitePredictorFree(
  JNIEnv* jenv, jclass jcls, jlong jhandle) {
  return (jint)0;
}
