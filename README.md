# quizScatterer

## 概要

日本語のクイズ問題の配置を最適化する（ジャンルで分散させる）ツールです。

## セットアップ

```shell
リポジトリコピー
git clone https://github.com/miyax0227/quizScatterer
# 必要なライブラリのインストール
cd quizScatterer
sudo pip3 install -r requirements.txt
# 学習済みモデルのダウンロード（白ヤギコーポレーション様）
sudo chmod 755 getGensimModel.sh
./getGensimModel.sh
```

学習済みモデルのダウンロードは，

```shell
make get_model
```

でもできます．

## コマンド

```shell
python3 -m quizscatterer INPUT_FILE (> OUTPUT_FILE)
```

例：

```shell
python3 -m quizscatterer sample.txt > result.txt
```

## `rye`を使ったセットアップ方法

パッケージ管理に`rye`を使うことができます．

### インストール

Mac/Linuxユーザは，

```shell
make install_rye
```

でインストールできます．

Windowsの場合は，[`rye`の公式サイト](https://rye.astral.sh/guide/installation/#installing-rye)からバイナリをインストールしてください．

### 必要なパッケージのインストール

```shell
make install
```

もしくは，

```shell
rye sync
```

でインストールできます．

### フォーマット

`ruff`を使ってコードをフォーマットできます．

```shell
make format
```

### テスト

`pytest`によるテストを実行するには，

```shell
make test
```

を実行してください．

### 実行コマンド

`rye`を使った場合は，`rye run`を最初につけて実行してください．

```shell
rye run python -m quizscatterer INPUT_FILE (> OUTPUT_FILE)
```

## 作者

Miyax ([@mi_yax](https://twitter.com/mi_yax))
