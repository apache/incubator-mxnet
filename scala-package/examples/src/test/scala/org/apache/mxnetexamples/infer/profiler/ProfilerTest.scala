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

package org.apache.mxnetexamples.profiler

import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory
import java.io.File

import org.apache.mxnet.Profiler
import org.apache.mxnet.Context

/**
  * Integration test for imageClassifier example.
  * This will run as a part of "make scalatest"
  */
class ProfilerTest extends FunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[ProfilerTest])

  test("testProfiler") {
    logger.info("Running profiler test...")

    val eray = new ProfilerNDArray
    try {

      val path = System.getProperty("java.io.tmpdir")
      val kwargs = Map("file_name" -> path)
      logger.info(s"profile file save to $path")

      Profiler.profilerSetState("run")
      ProfilerNDArray.testBroadcast()
      ProfilerNDArray.testNDArraySaveload()
      ProfilerNDArray.testNDArrayCopy()
      ProfilerNDArray.testNDArrayNegate()
      ProfilerNDArray.testNDArrayScalar()
      ProfilerNDArray.testClip()
      ProfilerNDArray.testDot()
      ProfilerNDArray.testNDArrayOnehot()
      Profiler.profilerSetState("stop")

    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        sys.exit(1)
      }
    }

  }
}
