#!/bin/bash
cd quizScatterer/gensimModel
if [ $? != 0 ]; then
  exit 255
fi
rm -f *
wget http://public.shiroyagi.s3.amazonaws.com/latest-ja-word2vec-gensim-model.zip
unzip latest-ja-word2vec-gensim-model.zip
rm latest-ja-word2vec-gensim-model.zip
exit 0
