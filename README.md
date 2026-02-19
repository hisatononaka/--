## 概要

- ハイパースペクトル画像（シーンデータ）向けの **DINO**（self-Distillation with NO labels）事前学習と、**単一ラベル分類**の下流タスクを行うプロジェクトです。
- 入力: HDF5 (.h5) 形式のハイパースペクトル画像。metadata（scene_tags / object_tags）を JSON で紐付けて利用できます。
- [Spectral Earth](https://github.com/AABNassim/spectral_earth) を参考にしています。

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
   `configs/data/scene.yaml` の `data_path`（デフォルト: `data/raw/scene`）に `.h5` / `.hdf5` を配置します。DataModule は配下を再帰的に検索して読みます。

2. **metadata（任意）**  
   シーン・オブジェクトタグを使う場合は、`metadata_path` で JSON を指定します。未指定時は `statistics_dir/metadata.json` を参照します。各エントリは `id`（H5 のキー等から導出した ID）と `scene_tags` / `object_tags` のリストを持ちます。

3. **統計量（正規化用）**  
   チャンネルごとの mean / std で正規化する場合は、事前に計算してください。

   ```bash
   python scripts/compute_statistics.py --data_path data/raw/scene --output_dir data/statistics
   ```

   出力: `mu.npy`, `sigma.npy`（DINOModule 等の NormalizeMeanStd で利用）。

4. **H5 → NPY 変換（任意）**  
   H5 を日付・グループ別の .npy に展開する場合:

   ```bash
   python scripts/h5_to_npy.py --data-dir data/raw/scene --output-dir data/npy
   ```

5. **データ・metadata の検証**  
   H5 と metadata の対応、単一ラベル絞り込み後のサンプル数、DataLoader の長さなどを確認する場合（DataLoader が空になる原因の切り分けにも利用）:

   ```bash
   python test.py
   ```

## 学習の実行

プロジェクトルートで実行します。設定は `configs/data/`, `configs/model/`, `configs/train.yaml` をマージして渡します。

### DINO 事前学習

```bash
python train.py
```

- `configs/data/scene.yaml` … データパス、バッチサイズ、statistics_dir、metadata_path、patch_size など
- `configs/model/dino.yaml` … バックボーン、global_size / local_size（crop サイズ）、dynamic_img_size（ViT 可変解像度）、multicrop / n_views
- `configs/train.yaml` … エポック数、ログディレクトリなど

チェックポイントは `log_dir` に、train_loss 最小のベストと最終エポックが保存されます。

### 単一ラベル分類

```bash
python train_singlelabel.py
```

- `configs/data/scene.yaml` で **`single_label_classes` を必ず指定**します（使用するラベル名のリスト。順序がクラス ID に対応）。
- `label_source`（`scene` / `object`）、`multi_to_single`（複数タグ時は `first`）、`max_samples` / `seed` でデータの絞り方を指定できます。
- `val_ratio` / `test_ratio` で train/val/test に分割（0 ならその分割なし）。学習後に test で最終精度を表示します。
- `configs/model/singlelabel_classification.yaml` でバックボーン、**`use_adapter`**（Spectral Earth と同様、下流でも SpectralAdapter を通す）、`linear_eval`、**`checkpoint_path`**（DINO 事前学習の student_backbone と spectral_adapter をロード）を指定します。`num_classes` は data の `single_label_classes` の長さから自動設定されます。入力画像は `patch_size`（data config）に合わせてリサイズされます。

## 設定例

| 設定                    | 説明                               | 例               |
| ----------------------- | ---------------------------------- | ---------------- |
| `data_path`             | H5 が入ったディレクトリ            | `data/raw/scene` |
| `statistics_dir`        | 正規化用 mu.npy / sigma.npy の場所 | `data/statistics/scene` |
| `metadata_path`         | scene_tags / object_tags 用 JSON   | （未指定時は statistics_dir/metadata.json） |
| `single_label_classes` | 単一ラベル分類で使うラベル名のリスト（必須） | `["class_a", "class_b"]` |
| `val_ratio` / `test_ratio` | 単一ラベル時の検証・テスト割合  | 0.1（val_ratio + test_ratio < 1） |
| `num_bands`             | スペクトルバンド数                  | 151              |
| `backbone_name`         | バックボーン                        | `vit_small`      |
| `use_adapter`           | 単一ラベルで SpectralAdapter を通すか（下流も adapter 通過） | `true`           |
| `patch_size`            | 空間サイズ（単一ラベル時のリサイズ先 / DINO の global_size に連動） | 128              |
| `global_size` / `local_size` | DINO の global / local crop サイズ（ViT は dynamic_img_size で両方受け付け可能） | 128 / 48         |
| `max_epochs`            | 学習エポック数                      | 100              |

詳細は各 YAML と `src/models/dino_module.py` / `src/models/singlelabel_classification_module.py` の docstring を参照してください。

## スクリプト

| スクリプト                      | 説明                                                                 |
| ------------------------------- | -------------------------------------------------------------------- |
| `train.py`                      | DINO 事前学習のエントリポイント                                      |
| `train_singlelabel.py`         | 単一ラベル分類のエントリポイント（single_label_classes 必須）       |
| `test.py`                       | データ・metadata の検証と DataLoader 空の原因特定（config / path / ID / 単一ラベル一致） |
| `scripts/compute_statistics.py` | H5 からチャンネルごとの mean / std を計算し mu.npy, sigma.npy に保存 |
| `scripts/h5_to_npy.py`         | H5 を日付・グループ別の .npy に展開（--data-dir, --output-dir）       |
| `scripts/check_h5.py`          | read_h5 の返り値（キー・アイテム・読み込み結果）を確認                |

## ディレクトリ構成

```
scene_downstream/
├── configs/
│   ├── data/scene.yaml                    # データ・DataModule（single_label 含む）
│   ├── model/dino.yaml                    # DINO 事前学習用
│   ├── model/singlelabel_classification.yaml  # 単一ラベル分類用
│   └── train.yaml                         # Trainer 共通設定
├── scripts/               # データ処理・統計計算・H5 確認
├── src/
│   ├── backbones/         # ViT, SpectralAdapter, レジストリ
│   ├── datamodules/       # SceneDataModule
│   ├── datasets/         # H5Dataset（metadata / 単一ラベル対応）
│   ├── models/           # DINOModule, SingleLabelClassificationModule
│   ├── transforms/       # NormalizeMeanStd
│   └── utils/            # H5 読み込み（read_h5）
├── train.py               # DINO 事前学習
├── train_singlelabel.py   # 単一ラベル分類
├── test.py                # H5 と metadata の検証
├── requirements.txt
└── README.md
```

## ライセンス・参照

- DINO: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- 実装では [Lightly](https://github.com/lightly-ai/lightly) の DINO 損失・投影ヘッド等を利用しています。
