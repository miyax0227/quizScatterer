#!/bin/bash
cd quizscatterer/gensim_model
if [ $? != 0 ]; then
  exit 255
fi
rm -f *
wget https://myx-quiz-public.s3.ap-northeast-1.amazonaws.com/latest-ja-word2vec-gensim-model.zip
unzip latest-ja-word2vec-gensim-model.zip
rm latest-ja-word2vec-gensim-model.zip
exit 0
