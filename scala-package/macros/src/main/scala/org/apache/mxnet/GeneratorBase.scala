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

import org.apache.mxnet.init.Base.{RefInt, RefLong, RefString, _LIB}
import org.apache.mxnet.utils.{CToScalaUtils, OperatorBuildUtils}

import scala.collection.mutable.ListBuffer
import scala.reflect.macros.blackbox

abstract class GeneratorBase {
  type Handle = Long

  case class Arg(argName: String, argType: String, argDesc: String, isOptional: Boolean) {
    def safeArgName: String = argName match {
      case "var" => "vari"
      case "type" => "typeOf"
      case _ => argName
    }
  }

  case class Func(name: String, desc: String, listOfArgs: List[Arg], returnType: String)

  def functionsToGenerate(isSymbol: Boolean, isContrib: Boolean): List[Func] = {
    val l = getBackEndFunctions(isSymbol)
    if (isContrib)
      l.filter(func => func.name.startsWith("_contrib_") || !func.name.startsWith("_"))
    else
      l.filterNot(_.name.startsWith("_"))
  }

  def typeSafeFunctionsToGenerate(isSymbol: Boolean, isContrib: Boolean): List[Func] = {
    // Operators that should not be generated
    val notGenerated = Set("Custom")

    val l = getBackEndFunctions(isSymbol)
    val res = if (isContrib)
      l.filter(func => func.name.startsWith("_contrib_") || !func.name.startsWith("_"))
    else
      l.filterNot(_.name.startsWith("_"))

    res.filterNot(ele => notGenerated.contains(ele.name))
  }

  protected def getBackEndFunctions(isSymbol: Boolean): List[Func] = {
    val opNames = ListBuffer.empty[String]
    _LIB.mxListAllOpNames(opNames)
    opNames.map(opName => {
      val opHandle = new RefLong
      _LIB.nnGetOpHandle(opName, opHandle)
      makeAtomicFunction(opHandle.value, opName, isSymbol)
    }).toList
  }

  private def makeAtomicFunction(handle: Handle, aliasName: String, isSymbol: Boolean): Func = {
    val name = new RefString
    val desc = new RefString
    val keyVarNumArgs = new RefString
    val numArgs = new RefInt
    val argNames = ListBuffer.empty[String]
    val argTypes = ListBuffer.empty[String]
    val argDescs = ListBuffer.empty[String]

    _LIB.mxSymbolGetAtomicSymbolInfo(
      handle, name, desc, numArgs, argNames, argTypes, argDescs, keyVarNumArgs)
    val paramStr = OperatorBuildUtils.ctypes2docstring(argNames, argTypes, argDescs)
    val extraDoc: String = if (keyVarNumArgs.value != null && keyVarNumArgs.value.length > 0) {
      s"This function support variable length of positional input (${keyVarNumArgs.value})."
    } else {
      ""
    }
    val realName = if (aliasName == name.value) "" else s"(a.k.a., ${name.value})"
    val docStr = s"$aliasName $realName\n${desc.value}\n\n$paramStr\n$extraDoc\n"
    // scalastyle:off println
    if (System.getenv("MXNET4J_PRINT_OP_DEF") != null
      && System.getenv("MXNET4J_PRINT_OP_DEF").toLowerCase == "true") {
      println("Function definition:\n" + docStr)
    }
    // scalastyle:on println
    val argList = argNames zip argTypes zip argDescs map { case ((argName, argType), argDesc) =>
      val family = if (isSymbol) "org.apache.mxnet.Symbol" else "org.apache.mxnet.NDArray"
      val typeAndOption =
        CToScalaUtils.argumentCleaner(argName, argType, family)
      Arg(argName, typeAndOption._1, argDesc, typeAndOption._2)
    }
    val returnType = if (isSymbol) "org.apache.mxnet.Symbol" else "org.apache.mxnet.NDArrayFuncReturn"
    Func(aliasName, desc.value, argList.toList, returnType)
  }

  /**
    * Generate class structure for all function APIs
    *
    * @param c
    * @param funcDef DefDef type of function definitions
    * @param annottees
    * @return
    */
  protected def structGeneration(c: blackbox.Context)
                                (funcDef: List[c.universe.DefDef], annottees: c.Expr[Any]*)
  : c.Expr[Any] = {
    import c.universe._
    val inputs = annottees.map(_.tree).toList
    // pattern match on the inputs
    val modDefs = inputs map {
      case ClassDef(mods, name, something, template) =>
        val q = template match {
          case Template(superMaybe, emptyValDef, defs) =>
            Template(superMaybe, emptyValDef, defs ++ funcDef)
          case ex =>
            throw new IllegalArgumentException(s"Invalid template: $ex")
        }
        ClassDef(mods, name, something, q)
      case ModuleDef(mods, name, template) =>
        val q = template match {
          case Template(superMaybe, emptyValDef, defs) =>
            Template(superMaybe, emptyValDef, defs ++ funcDef)
          case ex =>
            throw new IllegalArgumentException(s"Invalid template: $ex")
        }
        ModuleDef(mods, name, q)
      case ex =>
        throw new IllegalArgumentException(s"Invalid macro input: $ex")
    }
    // wrap the result up in an Expr, and return it
    val result = c.Expr(Block(modDefs, Literal(Constant())))
    result
  }

  protected def typedFunctionCommonArgDef(func: Func): List[String] = {
    // build function argument definition, with optionality, and safe names
    func.listOfArgs.map(arg =>
      if (arg.isOptional) {
        // let's avoid a stupid Option[Array[...]]
        if (arg.argType.startsWith("Array[")) {
          s"${arg.safeArgName} : ${arg.argType} = Array.empty"
        } else {
          s"${arg.safeArgName} : Option[${arg.argType}] = None"
        }
      }
      else {
        s"${arg.safeArgName} : ${arg.argType}"
      }
    )
  }
}
