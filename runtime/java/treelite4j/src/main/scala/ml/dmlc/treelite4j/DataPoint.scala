/*!
 * Copyright (c) 2019-2020 by Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.treelite4j

/**
  * A data point (instance)
  *
  * @param indices Feature indices of this point or `null` if the data is dense
  * @param values Feature values of this point
  */
case class DataPoint(
    indices: Array[Int],
    values: Array[Float]) extends Serializable {
  require(indices == null || indices.length == values.length, "indices and values must have the same number of elements")
}
