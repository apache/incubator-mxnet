(defproject hello "0.1-SNAPSHOT"
  :plugins [[lein-jupyter "0.1.16"]]
  :auto {:default {:file-pattern #"\.(clj)$"}}
  :main bi-lstm-sort.core
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [org.apache.mxnet.contrib.clojure/clojure-mxnet "1.5.0-SNAPSHOT"]])
