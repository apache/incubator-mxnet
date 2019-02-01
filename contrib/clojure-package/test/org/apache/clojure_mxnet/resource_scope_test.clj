;;
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

(ns org.apache.clojure-mxnet.resource-scope-test
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.resource-scope :as resource-scope]
            [clojure.test :refer :all]))


(deftest test-resource-scope-with-ndarray
  (let [native-resources (atom {})
        x (ndarray/ones [2 2])
        return-val (resource-scope/using
                    (let [temp-x (ndarray/ones [3 1])
                          temp-y (ndarray/ones [3 1])]
                      (swap! native-resources assoc :temp-x temp-x)
                      (swap! native-resources assoc :temp-y temp-y)
                      (ndarray/+ temp-x 1)))]
    (is (true? (ndarray/is-disposed (:temp-x @native-resources))))
    (is (true? (ndarray/is-disposed (:temp-y @native-resources))))
    (is (false? (ndarray/is-disposed return-val)))
    (is (false? (ndarray/is-disposed x)))
    (is (= [2.0 2.0 2.0] (ndarray/->vec return-val)))))

(deftest test-nested-resource-scope-with-ndarray
  (let [native-resources (atom {})
        x (ndarray/ones [2 2])
        return-val (resource-scope/using
                    (let [temp-x (ndarray/ones [3 1])]
                      (swap! native-resources assoc :temp-x temp-x)
                     (resource-scope/using
                      (let [temp-y (ndarray/ones [3 1])]
                        (swap! native-resources assoc :temp-y temp-y)))))]
    (is (true? (ndarray/is-disposed (:temp-y @native-resources))))
    (is (true? (ndarray/is-disposed (:temp-x @native-resources))))
    (is (false? (ndarray/is-disposed x)))))

(deftest test-resource-scope-with-sym
  (let [native-resources (atom {})
        x (sym/ones [2 2])
        return-val (resource-scope/using
                    (let [temp-x (sym/ones [3 1])
                          temp-y (sym/ones [3 1])]
                      (swap! native-resources assoc :temp-x temp-x)
                      (swap! native-resources assoc :temp-y temp-y)
                      (sym/+ temp-x 1)))]
    (is (true? (sym/is-disposed (:temp-x @native-resources))))
    (is (true? (sym/is-disposed (:temp-y @native-resources))))
    (is (false? (sym/is-disposed return-val)))
    (is (false? (sym/is-disposed x)))))

(deftest test-nested-resource-scope-with-ndarray
  (let [native-resources (atom {})
        x (ndarray/ones [2 2])
        return-val (resource-scope/using
                    (let [temp-x (ndarray/ones [3 1])]
                      (swap! native-resources assoc :temp-x temp-x)
                     (resource-scope/using
                      (let [temp-y (ndarray/ones [3 1])]
                        (swap! native-resources assoc :temp-y temp-y)))))]
    (is (true? (ndarray/is-disposed (:temp-y @native-resources))))
    (is (true? (ndarray/is-disposed (:temp-x @native-resources))))
    (is (false? (ndarray/is-disposed x)))))

(deftest test-nested-resource-scope-with-sym
  (let [native-resources (atom {})
        x (sym/ones [2 2])
        return-val (resource-scope/using
                    (let [temp-x (sym/ones [3 1])]
                      (swap! native-resources assoc :temp-x temp-x)
                     (resource-scope/using
                      (let [temp-y (sym/ones [3 1])]
                        (swap! native-resources assoc :temp-y temp-y)))))]
    (is (true? (sym/is-disposed (:temp-y @native-resources))))
    (is (true? (sym/is-disposed (:temp-x @native-resources))))
    (is (false? (sym/is-disposed x)))))

;;; Note that if first is returned the rest of the collection ndarrays will
;;; NOT be disposed
(deftest test-list-creation-with-returning-first
  (let [native-resources (atom {})
        return-val (resource-scope/using
                    (let [temp-ndarrays (mapv (constantly (ndarray/ones [3 1])) (range 5))
                          _ (swap! native-resources assoc :temp-ndarrays temp-ndarrays)]
                      (first temp-ndarrays)))]
    (is (false? (ndarray/is-disposed return-val)))
    (is (every? false? (mapv ndarray/is-disposed (:temp-ndarrays @native-resources))))))

;;; To have the rest of the collection disposed you need to call copy on the result
(deftest test-list-creation-with-return-a-copy-of-first
  (let [native-resources (atom {})
        return-val (resource-scope/using
                    (let [temp-ndarrays (repeatedly 3 #(ndarray/ones [3 1]))
                          _ (swap! native-resources assoc :temp-ndarrays temp-ndarrays)]
                      (ndarray/copy (first temp-ndarrays))))]
    (is (false? (ndarray/is-disposed return-val)))
    (is (every? true? (mapv ndarray/is-disposed (:temp-ndarrays @native-resources))))))

;;; If you don't assign them all to a let vector binding and just return the first
;; then they are disposed as usual
(deftest test-list-creation-with-return-a-copy-of-first
  (let [native-resources (atom [])
        return-val (resource-scope/using
                    (first (repeatedly 5 #(do
                                            (let [x (ndarray/ones [3 1])]
                                              (swap! native-resources conj x)
                                              x)))))]
    (is (false? (ndarray/is-disposed return-val)))
    (is (every? true? (mapv ndarray/is-disposed (:temp-ndarrays @native-resources))))))

(deftest test-with-let
  (let [native-resources (atom {})
        x (ndarray/ones [2 2])
        return-val (resource-scope/with-let [temp-x (ndarray/ones [3 1])
                                             temp-y (ndarray/ones [3 1])]
                     (swap! native-resources assoc :temp-x temp-x)
                     (swap! native-resources assoc :temp-y temp-y)
                     (ndarray/+ temp-x 1))]
    (is (true? (ndarray/is-disposed (:temp-x @native-resources))))
    (is (true? (ndarray/is-disposed (:temp-y @native-resources))))
    (is (false? (ndarray/is-disposed return-val)))
    (is (false? (ndarray/is-disposed x)))
    (is (= [2.0 2.0 2.0] (ndarray/->vec return-val)))))

(deftest test-with-do
  (let [native-resources (atom {})
        x (ndarray/ones [2 2])
        return-val (resource-scope/with-do
                     (swap! native-resources assoc :temp-x (ndarray/ones [3 1]))
                     (swap! native-resources assoc :temp-y (ndarray/ones [3 1]))
                     (ndarray/ones [3 1]))]
    (is (true? (ndarray/is-disposed (:temp-x @native-resources))))
    (is (true? (ndarray/is-disposed (:temp-y @native-resources))))
    (is (false? (ndarray/is-disposed return-val)))
    (is (false? (ndarray/is-disposed x)))
    (is (= [1.0 1.0 1.0] (ndarray/->vec return-val)))))
