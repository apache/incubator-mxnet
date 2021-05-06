# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# download the IMDB dataset
if (!file.exists("data/aclImdb_v1.tar.gz")) {
  download.file("https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
                "data/aclImdb_v1.tar.gz")
  untar("data/aclImdb_v1.tar.gz", exdir = "data/")
}

# install required packages
list.of.packages <- c("readr", "dplyr", "stringr", "stringi")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[, "Package"])]
if (length(new.packages)) install.packages(new.packages)

require("readr")
require("dplyr")
require("stringr")
require("stringi")

negative_train_list <- list.files("data/aclImdb/train/neg/", full.names = T)
positive_train_list <- list.files("data/aclImdb/train/pos/", full.names = T)

negative_test_list <- list.files("data/aclImdb/test/neg/", full.names = T)
positive_test_list <- list.files("data/aclImdb/test/pos/", full.names = T)

file_import <- function(file_list) {
  import <- sapply(file_list, read_file)
  return(import)
}

negative_train_raw <- file_import(negative_train_list)
positive_train_raw <- file_import(positive_train_list)

negative_test_raw <- file_import(negative_test_list)
positive_test_raw <- file_import(positive_test_list)

train_raw <- c(negative_train_raw, positive_train_raw)
test_raw <- c(negative_test_raw, positive_test_raw)

# Pre-process a corpus composed of a vector of sequences Build a dictionnary
# removing too rare words
text_pre_process <- function(corpus, count_threshold = 10, dic = NULL) {
  raw_vec <- corpus
  raw_vec <- stri_enc_toascii(str = raw_vec)
  
  ### perform some preprocessing
  raw_vec <- str_replace_all(string = raw_vec, pattern = "[^[:print:]]", replacement = "")
  raw_vec <- str_to_lower(string = raw_vec)
  raw_vec <- str_replace_all(string = raw_vec, pattern = "_", replacement = " ")
  raw_vec <- str_replace_all(string = raw_vec, pattern = "\\bbr\\b", replacement = "")
  raw_vec <- str_replace_all(string = raw_vec, pattern = "\\s+", replacement = " ")
  raw_vec <- str_trim(string = raw_vec)
  
  ### Split raw sequence vectors into lists of word vectors (one list element per
  ### sequence)
  word_vec_list <- stri_split_boundaries(raw_vec, type = "word", skip_word_none = T, 
    skip_word_number = F, simplify = F)
  
  ### Build vocabulary
  if (is.null(dic)) {
    word_vec_unlist <- unlist(word_vec_list)
    word_vec_table <- sort(table(word_vec_unlist), decreasing = T)
    word_cutoff <- which.max(word_vec_table < count_threshold)
    word_keep <- names(word_vec_table)[1:(word_cutoff - 1)]
    stopwords <- c(letters, "an", "the", "br")
    word_keep <- setdiff(word_keep, stopwords)
  } else word_keep <- names(dic)[!dic == 0]
  
  ### Clean the sentences to keep only the curated list of words
  word_vec_list <- lapply(word_vec_list, function(x) x[x %in% word_keep])
  
  # sentence_vec<- stri_split_boundaries(raw_vec, type='sentence', simplify = T)
  word_vec_length <- lapply(word_vec_list, length) %>% unlist()
  
  ### Build dictionnary
  dic <- 1:length(word_keep)
  names(dic) <- word_keep
  dic <- c(`¤` = 0, dic)
  
  ### reverse dictionnary
  rev_dic <- names(dic)
  names(rev_dic) <- dic
  
  return(list(word_vec_list = word_vec_list, dic = dic, rev_dic = rev_dic))
}

################################################################ 
make_bucket_data <- function(word_vec_list, labels, dic, seq_len = c(225), right_pad = T) {
  ### Trunc sequence to max bucket length
  word_vec_list <- lapply(word_vec_list, head, n = max(seq_len))
  
  word_vec_length <- lapply(word_vec_list, length) %>% unlist()
  bucketID <- cut(word_vec_length, breaks = c(0, seq_len, Inf), include.lowest = T, 
    labels = F)
  
  ### Right or Left side Padding Pad sequences to their bucket length with
  ### dictionnary 0-label
  word_vec_list_pad <- lapply(1:length(word_vec_list), function(x) {
    length(word_vec_list[[x]]) <- seq_len[bucketID[x]]
    word_vec_list[[x]][is.na(word_vec_list[[x]])] <- names(dic[1])
    if (right_pad == F) 
      word_vec_list[[x]] <- rev(word_vec_list[[x]])
    return(word_vec_list[[x]])
  })
  
  ### Assign sequences to buckets and unroll them in order to be reshaped into arrays
  unrolled_arrays <- lapply(1:length(seq_len), function(x) unlist(word_vec_list_pad[bucketID == 
    x]))
  
  ### Assign labels to their buckets
  bucketed_labels <- lapply(1:length(seq_len), function(x) labels[bucketID == x])
  names(bucketed_labels) <- as.character(seq_len)
  
  ### Assign the dictionnary to each bucket terms
  unrolled_arrays_dic <- lapply(1:length(seq_len), function(x) dic[unrolled_arrays[[x]]])
  
  # Reshape into arrays having each sequence into a row
  features <- lapply(1:length(seq_len), function(x) {
    array(unrolled_arrays_dic[[x]], 
          dim = c(seq_len[x], length(unrolled_arrays_dic[[x]])/seq_len[x]))
  })
  
  names(features) <- as.character(seq_len)
  
  ### Combine data and labels into buckets
  buckets <- lapply(1:length(seq_len), function(x) c(list(data = features[[x]]), 
    list(label = bucketed_labels[[x]])))
  names(buckets) <- as.character(seq_len)
  
  ### reverse dictionnary
  rev_dic <- names(dic)
  names(rev_dic) <- dic
  
  return(list(buckets = buckets, dic = dic, rev_dic = rev_dic))
}


corpus_preprocessed_train <- text_pre_process(corpus = train_raw, count_threshold = 10, 
  dic = NULL)

corpus_preprocessed_test <- text_pre_process(corpus = test_raw, dic = corpus_preprocessed_train$dic)

seq_length_dist <- unlist(lapply(corpus_preprocessed_train$word_vec_list, length))
quantile(seq_length_dist, 0:20/20)

# Save bucketed corpus
corpus_bucketed_train <- make_bucket_data(word_vec_list = corpus_preprocessed_train$word_vec_list, 
                                          labels = rep(0:1, each = 12500), 
                                          dic = corpus_preprocessed_train$dic, 
                                          seq_len = c(100, 150, 250, 400, 600), 
                                          right_pad = T)

corpus_bucketed_test <- make_bucket_data(word_vec_list = corpus_preprocessed_test$word_vec_list, 
                                         labels = rep(0:1, each = 12500), 
                                         dic = corpus_preprocessed_test$dic, 
                                         seq_len = c(100, 150, 250, 400, 600), 
                                         right_pad = T)

saveRDS(corpus_bucketed_train, file = "data/corpus_bucketed_train.rds")
saveRDS(corpus_bucketed_test, file = "data/corpus_bucketed_test.rds")

# Save non bucketed corpus
corpus_single_train <- make_bucket_data(word_vec_list = corpus_preprocessed_train$word_vec_list, 
                                          labels = rep(0:1, each = 12500), 
                                          dic = corpus_preprocessed_train$dic, 
                                          seq_len = c(600), 
                                          right_pad = T)

corpus_single_test <- make_bucket_data(word_vec_list = corpus_preprocessed_test$word_vec_list, 
                                         labels = rep(0:1, each = 12500), 
                                         dic = corpus_preprocessed_test$dic, 
                                         seq_len = c(600), 
                                         right_pad = T)

saveRDS(corpus_single_train, file = "data/corpus_single_train.rds")
saveRDS(corpus_single_test, file = "data/corpus_single_test.rds")
