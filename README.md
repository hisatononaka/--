## 概要

- ハイパースペクトル画像（シーンデータ）向けの **DINO**（self-Distillation with NO labels）事前学習プロジェクト
- DINOを用いた事前学習と下流タスクの性能評価
- Spectral Earthのgithubを参考にして，作成しました（URL:https://github.com/AABNassim/spectral_earth）
- 入力: HDF5 (.h5) 形式のハイパースペクトル画像

## 必要な環境

| 項目   | 推奨                                       |
| ------ | ------------------------------------------ |
| Python | 3.11.9                                     |
| GPU    | 任意                                       |
| メモリ | 8GB 以上（バッチサイズ・データ量に応じて） |

## セットアップ

```bash
# リポジトリのクローン
git clone <リポジトリURL> scene_downstream
cd scene_downstream

# 仮想環境（任意）
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

## データの準備

1. **HDF5 データ**  
   `configs/data/scene.yaml` の `data_path` で指定したディレクトリ（デフォルト: `data/raw/scene`）に `.h5` / `.hdf5` を配置します。  
   DataModule はその配下を再帰的に検索して H5 ファイルを読みます。

2. **統計量（正規化用）**  
   学習時にチャンネルごとの mean / std で正規化する場合は、事前に統計量を計算してください。

   ```bash
   python scripts/compute_statistics.py --data_path data/raw/scene --output_dir data/statistics
   ```

   出力: `data/statistics/mu.npy`, `data/statistics/sigma.npy`（DINO モジュールの NormalizeMeanStd で利用）。

3. **H5 → NPY 変換（任意）**  
   H5 を日付・グループ別の .npy に展開する場合は:

   ```bash
   python scripts/h5_to_npy.py --input_dir data/raw/scene --output_dir data/npy
   ```

## 学習の実行

プロジェクトルートで:

```bash
python train.py
```

設定は次の YAML から:

- `configs/data/scene.yaml` … データパス、バッチサイズ、パッチサイズなど
- `configs/model/dino.yaml` … バックボーン、DINO のハイパーパラメータ
- `configs/train.yaml` … エポック数、ログディレクトリなど

ログとチェックポイントは `logs/`（または `configs/train.yaml` の `log_dir`）に保存されます。  
`train_loss` 最小のベストモデルと最終エポックが保存されます。

## 設定例

| 設定            | 説明                    | 例               |
| --------------- | ----------------------- | ---------------- |
| `data_path`     | H5 が入ったディレクトリ | `data/raw/scene` |
| `num_bands`     | スペクトルバンド数      | 151              |
| `batch_size`    | バッチサイズ            | 4                |
| `patch_size`    | 入力パッチのサイズ      | 128              |
| `backbone_name` | バックボーン            | `vit_small`      |
| `max_epochs`    | 学習エポック数          | 100              |

詳細は各 YAML と `src/models/dino_module.py` の docstring を参照してください。

## スクリプト

| スクリプト                      | 説明                                                                  |
| ------------------------------- | --------------------------------------------------------------------- |
| `scripts/compute_statistics.py` | H5 からチャンネルごとの mean / std を計算し `data/statistics/` に保存 |
| `scripts/h5_to_npy.py`          | H5 を日付・グループ別の .npy に展開                                   |
| `scripts/check_h5.py`           | H5 ファイルの内容確認                                                 |
| `scripts/setup_vm.sh`           | VM 用のセットアップスクリプト                                         |

## ディレクトリ構成

```
scene_downstream/
├── configs/
│   ├── data/scene.yaml    # データ・DataModule 用設定
│   ├── model/dino.yaml    # DINO モデル用設定
│   └── train.yaml         # Trainer 用設定
├── docs/
│   └── VM_SETUP.md        # VM 環境構築手順
├── scripts/               # データ処理・統計計算など
├── src/
│   ├── backbones/         # ViT, SpectralAdapter など
│   ├── datamodules/       # Lightning DataModule（Scene）
│   ├── datasets/          # H5 データセット
│   ├── models/            # DINOModule
│   ├── transforms/        # 正規化など
│   └── utils/             # H5 読み込みなど
├── train.py               # 学習エントリポイント
├── requirements.txt
└── README.md
```

## ライセンス・参照

- DINO: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- 実装では [Lightly](https://github.com/lightly-ai/lightly) の DINO 損失・投影ヘッド等を利用しています。
