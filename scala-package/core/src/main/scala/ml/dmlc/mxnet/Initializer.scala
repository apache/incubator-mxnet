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

package ml.dmlc.mxnet

/**
 *
 * Base class for Initializer.
 */
abstract class Initializer {

  /**
   * Initialize an Initializer
   *
   * @param name name of corrosponding ndarray
   * @param arr ndarray to be Initialized
   */
  def apply(name: String, arr: NDArray): Unit = {

    if (name.startsWith("upsampling")) {
      initBilinear(name, arr)
    } else if (name.startsWith("stn_loc") && name.endsWith("weight")) {
      initZero(name, arr)
    } else if (name.startsWith("stn_loc") && name.endsWith("bias")) {
      initLocBias(name, arr)
    } else if (name.endsWith("bias")) {
      initBias(name, arr)
    } else if (name.endsWith("gamma")) {
      initGamma(name, arr)
    } else if (name.endsWith("beta")) {
      initBeta(name, arr)
    } else if (name.endsWith("weight")) {
      initWeight(name, arr)
    } else if (name.endsWith("moving_mean")) {
      initZero(name, arr)
    } else if (name.endsWith("moving_var")) {
      initZero(name, arr)
    } else if (name.endsWith("moving_inv_var")) {
      initZero(name, arr)
    } else if (name.endsWith("moving_avg")) {
      initZero(name, arr)
    } else {
      initDefault(name, arr)
    }
  }

  protected def initBilinear(name: String, arr: NDArray): Unit = {
    val weight = Array.fill[Float](arr.size)(0.0f)
    val shape = arr.shape
    val f = shape(3) / 2.0f
    val c = (2 * f - 1 - f % 2) / (2.0f * f)

    (0 until arr.size).foreach { i =>
      val x = i % shape(3)
      val y = (i / shape(3)) % shape(2)
      weight(i) = (1 - math.abs(x / f - c)) * (1 - math.abs(y / f - c))
    }

    arr.set(NDArray.array(weight, shape))
  }

  protected def initLocBias(name: String, arr: NDArray): Unit = {
    val shape = arr.shape
    require(shape(0) == 6)
    arr.set(Array(1f, 0f, 0f, 0f, 1f, 0f))
  }

  protected def initZero(name: String, arr: NDArray): Unit = {
    arr.set(0f)
  }

  protected def initBias(name: String, arr: NDArray): Unit = {
    arr.set(0f)
  }

  protected def initGamma(name: String, arr: NDArray): Unit = {
    arr.set(1f)
  }

  protected def initBeta(name: String, arr: NDArray): Unit = {
    arr.set(0f)
  }

  protected def initWeight(name: String, arr: NDArray): Unit

  protected def initDefault(name: String, arr: NDArray): Unit = {
    throw new IllegalArgumentException(s"Unknown initialization pattern for $name.")
  }
}

/**
 * Initialize the weight with mixed Initializer
 *
 * @param patterns List of regular expression patterns to match parameter names.
 * @param initializers List of Initializer corrosponding to patterns
 */
class Mixed(protected val patterns: List[String],
    protected val initializers: List[Initializer]) extends Initializer {
  require(patterns.length == initializers.length)
  private val map = patterns.map(_.r).zip(initializers)

  override def apply(name: String, arr: NDArray): Unit = {
    val matchR = map.filter { case (prog, init) => prog.findFirstIn(name) != None }
    if (matchR.length == 0) {
      throw new IllegalArgumentException(
          s"Parameter $name did not match any pattern. Consider " +
          "add a \".*\" pattern at the and with default Initializer.")
    } else matchR(0)._2(name, arr)
  }

  override def initWeight(name: String, arr: NDArray): Unit = {}
}

/**
 * Initializes weights to zero.
 */
class Zero extends Initializer {
  override def initWeight(name: String, arr: NDArray): Unit = {
    arr.set(0f)
  }
}

/**
 * Initializes weights to one.
 */
class One extends Initializer {
  override def initWeight(name: String, arr: NDArray): Unit = {
    arr.set(1f)
  }
}

/**
 * Initializes the weights to a scalar value.
 *
 * @param value The Fill value
 */
class Constant(protected val value: Float) extends Initializer {
  override def initWeight(name: String, arr: NDArray): Unit = {
    arr.set(value)
  }
}

/**
 * Initialize the weight with uniform [-scale, scale]
 *
 * @param scale The scale of uniform distribution
 */
class Uniform(protected val scale: Float = 0.07f) extends Initializer {
  override def initWeight(name: String, arr: NDArray): Unit = {
    Random.uniform(-scale, scale, out = arr)
  }
}


/**
 * Initialize the weight with normal(0, sigma)
 *
 * @param sigma Standard deviation for gaussian distribution.
 */
class Normal(protected val sigma: Float = 0.01f) extends Initializer {
  override def initWeight(name: String, arr: NDArray): Unit = {
    Random.normal(0, sigma, out = arr)
  }
}


/**
 * Initialize the weight with Xavier or similar initialization scheme.
 *
 * @param rndType Options are: "gaussian" or "uniform"
 * @param factorType Options are: "avg", "in", "out"
 * @param magnitude scale of random number range
 */
class Xavier(protected val rndType: String = "uniform",
             protected val factorType: String = "avg",
             protected val magnitude: Float = 3) extends Initializer {

  override def initWeight(name: String, arr: NDArray): Unit = {
    val shape = arr.shape
    val fanIn = shape.slice(1, shape.length).product
    val fanOut = shape(0)
    var factor = 1f

    factor = factorType match {
      case "avg" => (fanIn + fanOut) / 2f
      case "in" => fanIn
      case "out" => fanOut
      case _ => throw new IllegalArgumentException("Incorrect factor type")
    }
    val scale = math.sqrt(magnitude / factor).toFloat

    rndType match {
      case "uniform" => Random.uniform(-scale, scale, out = arr)
      case "gaussian" => Random.normal(0, scale, out = arr)
      case _ => throw new IllegalArgumentException("Unknown random type")
    }
  }
}

/**
 * Initialize the weight according to a MSRA paper.
 *
 * This initializer implements *Delving Deep into Rectifiers: Surpassing
 * Human-Level Performance on ImageNet Classification*, available at
 * https://arxiv.org/abs/1502.01852.
 *
 * This initializer is proposed for initialization related to ReLu activation,
 * it maked some changes on top of Xavier method.
 *
 * @param factorType Options are: "avg", "in", "out"
 * @param slop Initial slope of any PReLU (or similar) nonlinearities.
 */
class MSRAPrelu(factorType: String = "avg", slope: Float = 0.25f) extends
  Xavier("gaussian", factorType, 2f / (1 + slope * slope)) {
}

/**
 * Initialize weight for upsampling layers.
 */
class Bilinear extends Initializer {
  override def initWeight(name: String, arr: NDArray): Unit = {
    val weight = Array.fill[Float](arr.shape.product)(0f)
    val shape = arr.shape
    val f = Math.ceil(shape(3) / 2f)
    val c = (2 * f - 1 - f % 2) / (2 * f)
    for (i <- 0 until shape.product) {
      val x = i % shape(3)
      val y = (i / shape(3)) % shape(2)
      weight(i) = ((1 - Math.abs(x / f - c)) * (1 - Math.abs(y / f - c))).toFloat
    }
    arr.set(weight)
  }
}

/**
 * Initialize all bias of an LSTMCell to 0.0 except for
 * the forget gate whose bias is set to custom value.
 *
 * @param forgetBias Bias for the forget gate.
 *                      Jozefowicz et al. 2015 recommends setting this to 1.0.
 */
class LSTMBias(protected val forgetBias: Float) extends Initializer {
  override def initWeight(name: String, arr: NDArray): Unit = {
    arr.set(0f)
    // in the case of LSTMCell the forget gate is the second
    // gate of the 4 LSTM gates, we modify the according values.
    val numHidden = arr.shape(0) / 4
    val tmp = arr.toArray
    for (i <- numHidden until numHidden * 2) tmp(i) = this.forgetBias
    arr.set(tmp)
  }
}
