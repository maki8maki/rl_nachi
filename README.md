# rl_nachi

マニピュレータを強化学習で動かすためのパッケージ

## 使い方

### 共通事項

事前にマニピュレータ用のパッケージやRGB-D画像取得用のパッケージをセットアップしておく

#### セットアップ

ワークスペースを作成し、レポジトリをクローンする

```bash
mkdir -p catkin_ws/src
cd ~/catkin_ws/src
git clone git@github.com:maki8maki/rl_nachi.git
```

必要なパッケージをインストールし、ビルドする

```bash
cd ~/catkin_ws
rosdep install -i --from-paths src/ -y
catkin build
```

config/**/*.yamlにパラメータを書き（書き方は各フォルダのdefaults.yamlを参照する）、実行ファイルで指定する

#### 実行

ターミナルA：マニピュレータとの通信を開始

```bash
cd ~/catkin_ws
source /opt/ros/<ros1_distro>/setup.bash
source devel/setup.bash
roslaunch rl_nachi nachi_bringup.launch
```

ターミナルB：RGB-D画像取得の実行

ターミナルC：データ収集・動作生成の実行

```bash
cd ~/catkin_ws
source /opt/ros/<ros1_distro>/setup.bash
source devel/setup.bash
rosrun rl_nachi hoge.py
```

### ドメイン適応用のデータ収集

env.pyの`SHIFT_MIN`をデータ収集用のものに変更し、`data_collection_*.py`を実行する

### 動作生成

env.pyの`SHIFT_MIN`を本動作用のものに変更し、`Z_DIFF`を適切な値に設定したうえで、`sb3_da.py`を実行する

## ファイル構成

* lauch/
  * lauchファイルを格納
* logs/
  * ログファイルを格納
* rviz/
  * rvizファイルを格納
* scripts/
  * メインのPythonファイルを格納
  * agents/
    * モデルの定義
    * [agentsリポジトリ](https://github.com/maki8maki/agents)をサブモジュールとしたフォルダ
    * [CycleGANリポジトリ](https://github.com/maki8maki/pytorch-CycleGAN-and-pix2pix.git)をサブモジュールとして持つ
  * config/
    * パラメータファイルを格納する
    * 直下には動作生成に関するパラメータを格納
    * da/
      * ドメイン適応モデルに関するパラメータ
    * fe/
      * 特徴量抽出モデルに関するパラメータ
  * data/
    * ドメイン適応の学習のために収集したデータが保存される
  * model/
    * 各モデルのパラメータファイルを配置する
  * data_collection_policy.py
    * 強化学習の方策にしたがって動作し、ドメイン適応の学習のためのデータを収集する
  * data_collection_random.py
    * ランダムに動作し、ドメイン適応の学習のためのデータを収集する
  * playback.py
    * ログにしたがって動作を再現する
  * sb3_da.py
    * ドメイン適応**あり**の動作
  * sb3.py
    * ドメイン適応**なし**の動作
* urdf/
  * マニピュレータのモデルファイルやURDFファイルを格納
