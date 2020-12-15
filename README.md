# quizScatterer
## 概要
日本語のクイズ問題の配置を最適化する（ジャンルで分散させる）ツールです。

## セットアップ

```shell
# リポジトリコピー
git clone https://github.com/miyax0227/quizScatterer
# 必要なライブラリのインストール
cd quizScatterer
sudo pip3 install -r requirements.txt
# 学習済みモデルのダウンロード（白ヤギコーポレーション様）
sudo chmod 755 getGensimModel.sh
./getGensimModel.sh
```

## コマンド

```shell
python3 -m quizScatterer INPUT_FILE (> OUTPUT_FILE)
```

例：
```shell
python3 -m quizScatterer sample.txt > result.txt
```

## 作者
Miyax ([@mi_yax](https://twitter.com/mi_yax))