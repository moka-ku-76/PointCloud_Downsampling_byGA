# 点群の特徴を考慮したダウンサンプリング


2022/02/19
2022/03/19更新

京都大学情報学研究科通信情報システム専攻

岡 誠道（おか　まさみち）


主に必要なライブラリ
1. [Pytorch](https://pytorch.org/)
2. [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
3. [DEAP](https://deap.readthedocs.io/en/master/)

参考サイト：[DEAPを用いた特徴選択](https://qiita.com/kimisyo/items/2a1fc6a28b389f3e0561)

その他
numpy, pandas, pathlibなどの基本ライブラリ

python バージョン 3.8.10で動作確認済み

また、実行ファイルは全てipynbファイルであり、jupyter notebook上での実行を想定している。
jupyter notebookの使い方については[こちら](https://ai-inter1.com/jupyter-notebook/)

# 準備
1. このディレクトリをクローン  
`git clone https://github.com/moka-ku-76/PointCloud_Downsampling_byGA`

2. PointNet++のディレクトリをクローン  
`git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch`

3. ~ModelNet40のデータセットを[こちら](https://www.kaggle.com/balraj98/modelnet40-princeton-3d-object-dataset)からダウンロード（Kaggleアカウント必要、サイズは約10GB）~
（追記）データセットをどこからダウンロードしたのか忘れました。[こちら](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)と同様に[ModelNet40ホームページ](https://modelnet.cs.princeton.edu/)からダウンロードした気がしますが、現在できなくなっています。（ダウンロードをクリックしても反応しない。）ホームページからデータセットをダウンロードできた人は次に進んでください。

4. `Pointnet_Pointnet2_pytorch/`下にdataディレクトリ作成し、解凍  
`mkdir Pointnet_Pointnet2_pytorch/data`  
`unzip [ダウンロードしたディレクトリ]/modelnet40_mormal_resampled.zip -d Pointnet_Pointnet2_pytorch/data`

5. [DEAP](https://deap.readthedocs.io/en/master/)のインストール  
`pip install deap`

# 実行手順
1. Base_featue_vector.ipynb
2. GA_downsampling.ipynb
3. (Graph.ipynb)
4. Downsampling_test.ipynb
5. (Graph.ipynb)

別の学習済みモデルを利用したいときはPointNet++の手順に従って、あらためて学習しチェックポイントを作成する。

## Shareフォルダの構成（全て実行後）

```
Share/
    ├── Mywork/
    │   ├── Base_featue_vector.ipynb
    │   ├── DataLoaderForGAdownsampling.py
    │   ├── Downsampling_test.ipynb
    │   ├── GA_downsampling.ipynb
    │   ├── Graph.ipynb
    │   ├── Results/
    │   │   ├── Figures/
    │   │   │   ├── GA_downsampling/
    │   │   │   └── Thesis/
    │   │   └── Files/
    │   │       ├── downsampling/
    │   │       └── seed0/
    │   │           ├── Feature_vector_base/
    │   │           ├── GA_downsampling/
    │   │           │   ├── Hall of fame/
    │   │           │   ├── log/
    │   │           │   └── pop/
    │   │           └── target/
    │   ├── __pycache__/
    │   └── cls_ssg_feature_selection.py
    ├── Pointnet_Pointnet2_pytorch/
    └── README.md
```


# 各ファイルの概要
**Base_featue_vector.ipynb**　　#目的関数として利用する、ベース特徴ベクトルを計算する

**DataLoaderForGAdownsampling.py**  #参考コードのPointnet_Pointnet2_pytorch/data_utils/ModelNetDataLoader.pyを遺伝アルゴリズム用に変更したもの。特徴ベクトルも同時に取り出せるようになっている。

**Downsampling_test.ipynb**  #ランダム、FPS、提案手法の性能を比較する

**GA_downsampling.ipynb**  #遺伝的アルゴリズムで探索を実行するファイル。最も時間がかかる

**Graph.ipynb**  #各ipynbファイルで保存したcsvファイルをもとに図を描画するためのファイル

**cls_ssg_feature_selection.py**  #機械学習のモデルが書かれたファイル。参考コードのPointnet_Pointnet2_pytorch/models/pointnet2_cls_ssg.pyを２つに分割させた。チェックポイントは同じものを利用することができる。


**Results**
**Figures**  #図を保存するディレクトリ。修論に利用した図はThesisにある。他は参考程度に。

**Files**  #実験で取得した結果をcsv形式で保存している

 **downsampling** #３つの手法のダウンサンプリングテストの結果を保存
 
 **GA_downsampling** #遺伝的アルゴリズムの結果を保存
 
  _Hall of fame_ #殿堂入り。最も適応値が高い個人を保存。0,1で保存
  
  _log_ #世代ごとの適応値と正解率の最大、最小、平均、標準偏差を保存。
  
  _pop_ #最終集団の情報を保存。個体数は30。選択インデックスを保存。
  
 **target** #正解ラベルのデータを保存。




```python

```
