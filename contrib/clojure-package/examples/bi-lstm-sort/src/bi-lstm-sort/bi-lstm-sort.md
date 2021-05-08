
# Sorting numbers using a Bi-LSTM

In this notebook, we will show you how to sort a sequence of integers using a bi-lstm.
It also shows how you can use the newer ndarray and symbol APIs in clojure to accomplish this.

Let's first start off with importing the libraries required.


```clojure
;; Licensed to the Apache Software Foundation (ASF) under one or more
;; contributor license agreements.  See the NOTICE file distributed with
;; this work for additional information regarding copyright ownership.
;; The ASF licenses this file to You under the Apache License, Version 2.0
;; (the "License"); you may not use this file except in compliance with
;; the License.  You may obtain a copy of the License at
;;
;;    http://www.apache.org/licenses/LICENSE-2.0
;;
;; Unless required by applicable law or agreed to in writing, software
;; distributed under the License is distributed on an "AS IS" BASIS,
;; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
;; See the License for the specific language governing permissions and
;; limitations under the License.
;;

(ns bi-lstm-sort.core
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.initializer :as initializer]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.lr-scheduler :as lr-scheduler]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.ndarray-api :as ndarray-api]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.symbol-api :as sym-api]
            [org.apache.clojure-mxnet.visualization :as viz]))
```

## Data Preparation


```clojure
;; (def max-num 999)
;; (def dataset-size 60000)
;; (def seq-len 5)
;; (def split 0.8)
;; (def batch-size 256)

(def max-num 99)
(def dataset-size 5000)
(def seq-len 5)
(def split 0.8)
(def batch-size 50)
```




    #'bi-lstm-sort.core/batch-size



We are getting a dataset of **dataset-size** sequences of integers of length **seq-len** between **0** and **max-num**. 
We use **split*100%** of them for training and the rest for testing.


For example:

50 10 20 99 30

Should return

10 20 30 50 99


```clojure
;; (def num-layers 2)
;; (def state-size 128)

(def num-layers 1)
(def state-size 64)

(def max-len (+ (* seq-len (count (str max-num)))
                (- seq-len 1)))
(println "Max sequence length:" max-len)
```

    Max sequence length: 14


Above we define the number of layers **num-layers** and the size of hidden states **state-size** we will be using in the LSTM. In addition, we need to specify the maximum length of the sequences since we will be converting the input sequence to a one-hot encoding.


```clojure
(def num->idx
  (->> (range 10)
       (mapv (fn [i] [(str i) i]))
       (into {" " 10})
       (mapv (fn [[k i]] [(.charAt k 0) i]))
       (into {})))
(def vocab-len (count num->idx))

(println "Number to index:" num->idx)
(println "Vocabulary size:" vocab-len)
```

    Number to index: {  10, 0 0, 1 1, 2 2, 3 3, 4 4, 5 5, 6 6, 7 7, 8 8, 9 9}
    Vocabulary size: 11


Now let's generate some data and look at the input and the output. We will use the newer `ndarray-api/sort` method to sort the integer sequence.


```clojure
(def X (-> (random/uniform 0 max-num [dataset-size seq-len])
           (ndarray/as-type dtype/INT32)))
(def Y (ndarray-api/sort {:data X}))
(println "Input:" (ndarray/->vec (ndarray/at X 0))
         "\nTarget:" (ndarray/->vec (ndarray/at Y 0)))
(println "shape(X):" (ndarray/shape-vec X) 
         "shape(Y):" (ndarray/shape-vec Y))
```

    Input: [54.0 58.0 70.0 83.0 59.0] 
    Target: [54.0 58.0 59.0 70.0 83.0]
    shape(X): [5000 5] shape(Y): [5000 5]



```clojure
(def split-idx (int (* dataset-size split)))
(println "Split @" split-idx)
```

    Split @ 4000



```clojure
(defn to-num-array
    "Converts the input to sequence of indices from the vocabulary"
    [v]
  (let [nums (mapv (comp str int) v)           ; Convert the integers to strings
        joined (clojure.string/join " " nums)  ; Concatenate the strings
        padding (apply str (repeat (- max-len (count joined)) " "))  ; Find the padding length up to max-len
        padded (str joined padding)            ; Pad the string
        indices (mapv num->idx padded)]        ; Find the vocabulary indices
    (ndarray/array indices [max-len])))

(defn transform-data
    "Converts the vocabulary indices to one-hot encoding for input data"
    [batch]
  (let [data (-> batch
                 mx-io/batch-data
                 first
                 ndarray/->nd-vec)
        data-indices (mapv to-num-array data)]
    (mapv #(ndarray-api/one-hot % vocab-len)
          data-indices)))
  
(defn transform-label
    "Converts the vocabulary indices to output label"
    [batch]
  (let [label (-> batch
                  mx-io/batch-label
                  first
                  ndarray/->nd-vec)
        label-indices (mapv to-num-array label)]
    label-indices))
```




    #'bi-lstm-sort.core/transform-label



## Data Iterator

The iterator method below does a few different things:
1. Slices the `X` and `Y` arrays to get the train or test split
2. Creates an iterator over them and transforms the batches using `transform-data` and `transform-label` methods
3. Flattens the transformed data and expands the first dimension
4. Defines the data descriptors for the data, label, hidden state and LSTM state cell
5. Finally defines the `ndarray-iter` that will feed into training


```clojure
(defn get-iterator [begin end]
  (let [Xt (ndarray-api/slice-axis {:data X :axis 0 :begin begin :end end})
        Yt (ndarray-api/slice-axis {:data Y :axis 0 :begin begin :end end})
        [num-examples seq-len] (ndarray/shape-vec Xt)
        num-iter (mx-io/ndarray-iter [Xt] {:label [Yt] :data-batch-size batch-size})
        init-size (* batch-size (quot (+ batch-size num-examples) batch-size))
        batched-data (mx-io/for-batches num-iter transform-data)
        batched-label (mx-io/for-batches num-iter transform-label)
        exp-data (->> batched-data
                      flatten
                      (mapv #(ndarray-api/expand-dims % 0)))
        exp-label (->> batched-label
                       flatten
                       (mapv #(ndarray-api/expand-dims % 0)))
        data-desc (mx-io/data-desc {:name "data"
                                    :shape [num-examples max-len vocab-len]
                                    :dtype dtype/FLOAT32
                                    :layout org.apache.clojure-mxnet.layout/NTC})
        init-h-desc (mx-io/data-desc {:name "LSTM_init_h"
                                      :shape [num-examples (* 2 num-layers) state-size]
                                      :dtype dtype/FLOAT32
                                      :layout org.apache.clojure-mxnet.layout/NTC})
        init-c-desc (mx-io/data-desc {:name "LSTM_init_c"
                                      :shape [num-examples (* 2 num-layers) state-size]
                                      :dtype dtype/FLOAT32
                                      :layout org.apache.clojure-mxnet.layout/NTC})
        label-desc (mx-io/data-desc {:name "label"
                                     :shape [num-examples max-len]
                                     :dtype dtype/FLOAT32
                                     :layout org.apache.clojure-mxnet.layout/NT})
        data (ndarray-api/concat {:data exp-data
                                  :num-args (count exp-data)
                                  :dim 0})
        init-h (ndarray/zeros [init-size (* 2 num-layers) state-size])
        init-c (ndarray/zeros [init-size (* 2 num-layers) state-size])
        label (ndarray-api/concat {:data exp-label
                                   :num-args (count exp-label)
                                   :dim 0})]
    (mx-io/ndarray-iter {data-desc data
                         init-h-desc init-h
                         init-c-desc init-c}
                        {:label {label-desc label}
                         :data-batch-size batch-size})))
```




    #'bi-lstm-sort.core/get-iterator




```clojure
(def train-iter (get-iterator 0 split-idx))
(def eval-iter (get-iterator (+ split-idx 1) dataset-size))
```




    #'bi-lstm-sort.core/eval-iter




```clojure
(def batch (mx-io/next train-iter))
(def x (mx-io/batch-data batch))
(def y (mx-io/batch-label batch))
(println (ndarray-api/swap-axis {:data (first x) :dim1 0 :dim2 1}))
(println (first x))
(println (first y))
```

    #object[org.apache.mxnet.NDArray 0x3b52093f [
     [
      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
     
      ... with length 50
     ]
     [
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
      [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
     
      ... with length 50
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
     
      ... with length 50
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
     
      ... with length 50
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
     
      ... with length 50
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
     
      ... with length 50
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
     
      ... with length 50
     ]
     [
      [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
     
      ... with length 50
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
     
      ... with length 50
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
     
      ... with length 50
     ]
    
     ... with length 14
    ]
    <NDArray (14,50,11) cpu(0) float32>]
    #object[org.apache.mxnet.NDArray 0xc2d04f7 [
     [
      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
     
      ... with length 14
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
     
      ... with length 14
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
     
      ... with length 14
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
     
      ... with length 14
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
     
      ... with length 14
     ]
     [
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
      [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
     
      ... with length 14
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
     
      ... with length 14
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
     
      ... with length 14
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
     
      ... with length 14
     ]
     [
      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
      [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
      [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
     
      ... with length 14
     ]
    
     ... with length 50
    ]
    <NDArray (50,14,11) cpu(0) float32>]
    #object[org.apache.mxnet.NDArray 0x3040fe20 [
     [5.0,4.0,10.0,5.0,8.0,10.0,5.0,9.0,10.0,7.0,0.0,10.0,8.0,3.0]
     [4.0,1.0,10.0,5.0,3.0,10.0,6.0,1.0,10.0,8.0,3.0,10.0,8.0,4.0]
     [2.0,9.0,10.0,3.0,8.0,10.0,4.0,3.0,10.0,6.0,3.0,10.0,8.0,8.0]
     [5.0,10.0,2.0,6.0,10.0,3.0,7.0,10.0,4.0,7.0,10.0,9.0,5.0,10.0]
     [4.0,7.0,10.0,5.0,2.0,10.0,5.0,6.0,10.0,7.0,8.0,10.0,8.0,0.0]
     [7.0,10.0,3.0,3.0,10.0,3.0,8.0,10.0,8.0,2.0,10.0,9.0,1.0,10.0]
     [2.0,10.0,8.0,10.0,3.0,6.0,10.0,6.0,4.0,10.0,8.0,2.0,10.0,10.0]
     [1.0,3.0,10.0,7.0,7.0,10.0,8.0,6.0,10.0,8.0,6.0,10.0,9.0,4.0]
     [4.0,5.0,10.0,4.0,6.0,10.0,7.0,9.0,10.0,7.0,9.0,10.0,9.0,6.0]
     [1.0,1.0,10.0,5.0,1.0,10.0,6.0,7.0,10.0,7.0,1.0,10.0,7.0,7.0]
    
     ... with length 50
    ]
    <NDArray (50,14) cpu(0) float32>]


The RNN supported by MXNet expects data in a time-major format instead of a batch-major one. This means that the first dimension is time-steps followed by batch-size and then input-size.


```clojure
(defn get-lstm-outputs []
    (let [data (sym/variable "data")
          ;; rnn inputs are time-major (time length, batch size, dimensions)
          rnn-h-init (sym-api/swap-axis {:name "init_h_tm"
                                         :data (sym/variable "LSTM_init_h")
                                         :dim1 0 :dim2 1})
          rnn-c-init (sym-api/swap-axis {:name "init_c_tm"
                                         :data (sym/variable "LSTM_init_c")
                                         :dim1 0 :dim2 1})
          rnn-params (sym/variable "LSTM_weight")
          data-tm (sym-api/swap-axis {:name "data_tm" :data data
                                      :dim1 0 :dim2 1})
          rnn-outputs (sym-api/rnn {:name "LSTM"
                                    :data data-tm
                                    :parameters rnn-params
                                    :num-layers num-layers
                                    :mode "lstm"
                                    :bidirectional true
                                    :state rnn-h-init
                                    :state-cell rnn-c-init
                                    :state-size state-size
                                    :state-outputs true})
          rnn-states (sym/get rnn-outputs 0)
          rnn-preds (sym-api/fully-connected {:name "predictions"
                                              :data rnn-states
                                              :num-hidden vocab-len
                                              :flatten false})]
      rnn-preds))

(defn get-label-outputs []
    (let [label (sym/variable "label")
          label-tm (sym-api/swap-axis {:name "label_tm" :data label
                                       :dim1 0 :dim2 1})]
        label-tm))
```




    #'bi-lstm-sort.core/get-label-outputs




```clojure
(defn softmax-outputs [rnn-preds labels]
    (sym-api/softmax-output {:name "softmax"
                             :data rnn-preds
                             :label labels
                             :preserve-shape true}))
```




    #'bi-lstm-sort.core/softmax-outputs



We can visualize the outputs of LSTM and label networks:


```clojure
(-> (get-lstm-outputs)
    (sym/infer-shape {"data" [5 19 11]
                      "LSTM_init_h" [5 2 64]
                      "LSTM_init_c" [5 2 64]}))
```




    (([5 19 11] [39424] [5 2 64] [5 2 64] [11 128] [11]) ([19 5 11]) ())




```clojure
(-> (get-label-outputs)
    (sym/infer-shape {"label" [5 19]}))
```




    (([5 19]) ([19 5]) ())




```clojure
(let [rnn-preds (get-lstm-outputs)
      labels (get-label-outputs)
      softmax-output (softmax-outputs rnn-preds labels)]
    (sym/infer-shape softmax-output {"data" [5 19 11]
                                     "label" [5 19]
                                     "LSTM_init_h" [5 2 64]
                                     "LSTM_init_c" [5 2 64]}))
```




    (([5 19 11] [39424] [5 2 64] [5 2 64] [11 128] [11] [5 19]) ([19 5 11]) ())



## Training and Evaluation

We can now define the custom accuracy method to evaluate the output of the label sequence.


```clojure
(defn get-best-predictions [pred-tm]
    (as-> pred-tm data
          (ndarray-api/swap-axis {:data data :dim1 1 :dim2 0})
          (ndarray-api/argmax {:data data :axis -1})))

(defn accuracy [label pred-tm]
    (let [best-pred (get-best-predictions pred-tm)
          matches (ndarray/equal label best-pred)
          num-matches (-> matches ndarray/sum ndarray/->vec first)
          size (-> matches ndarray/size)]
        (float (/ num-matches size))))
```




    #'bi-lstm-sort.core/accuracy



### Note that the evaluation metrics get logged to the console


```clojure
(def model
    (let [rnn-preds (get-lstm-outputs)
          labels (get-label-outputs)
          softmax-output (softmax-outputs rnn-preds labels)
          model (m/module softmax-output {:data-names ["data" "LSTM_init_h" "LSTM_init_c"]
                                          :label-names ["label"]})
          schedule (lr-scheduler/factor-scheduler (* 1 dataset-size) 0.75)]
        (m/fit model {:train-data train-iter
                      :eval-data eval-iter
                      :num-epoch 200
                      :fit-params (m/fit-params {:optimizer (optimizer/adam {:learning-rate 0.02
                                                                             ;:lr-scheduler schedule
                                                                             })
                                                 :batch-end-callback (callback/speedometer batch-size 10)
                                                 :initializer (initializer/xavier {:factor-type "in" :magnitude 2.34})
                                                 :eval-metric (org.apache.clojure-mxnet.eval-metric/custom-metric
                                                                  #(accuracy %1 %2)
                                                                  "accuracy")})})))
```




    #'bi-lstm-sort.core/model



## Testing

Let's pick a random example from the test data.


```clojure
(def offset (rand-int (- dataset-size split-idx)))
(def x-orig (ndarray/at X (+ split-idx offset)))
(def y-orig (ndarray/at Y (+ split-idx offset)))

(println "X:" x-orig "\nY:" y-orig)
```

    X: #object[org.apache.mxnet.NDArray 0x27a19e2e [22.0,7.0,82.0,3.0,14.0]
    <NDArray (5) cpu(0) int32>] 
    Y: #object[org.apache.mxnet.NDArray 0x55bbe103 [3.0,7.0,14.0,22.0,82.0]
    <NDArray (5) cpu(0) int32>]



```clojure
(def x (-> x-orig
           ndarray/->vec
           to-num-array
           (ndarray-api/one-hot vocab-len)))
(def y (-> y-orig
           ndarray/->vec
           to-num-array))

```




    #object[org.apache.mxnet.NDArray 0x490e5f63 "[\n [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]\n [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]\n [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]\n [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]\n [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]\n [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]\n [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]\n [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]\n [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]\n [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]\n\n ... with length 14\n]\n<NDArray (14,11) cpu(0) float32>"]


