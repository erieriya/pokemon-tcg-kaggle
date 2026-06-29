# Kaggle調査ノート(リーダーボード・メタ・ロードマップ)

調査日: 2026-06-30

このファイルは外部(Kaggle)情報のスナップショット。スコア・メタは時間とともに変化するため、
古くなったら更新するか「調査日」を見て鮮度を判断すること。

## 1. 最重要発見: 本番ラダースコアがローカルテストと一致しない

公式リーダーボード(`pokemon-tcg-ai-battle`、3740チーム)で自チーム(Eriya / haaaal)を確認:

- **現在の順位: 3604位 / 3740(下位3.6%)、スコア285.1**
- 上位: 1位1469.4、上位20位は1197.4〜1469.4、中央値674.8
- 自分たちの提出履歴(`kaggle competitions submissions`、新しい順):

| 提出内容 | publicScore |
|---|---|
| heuristic v2 (dragapult_agent_v2相当、Crustle対策込み) | **164.7** |
| PPO rebalanced 2000ep | 285.1 |
| debug_v2 (素のヒューリスティック) | 289.6 |
| **DEBUG: ランダムエージェント+自分のデッキ** | **595.7** |
| **公式サンプル提出(素のランダムエージェント)** | **532.7** |

**何もしない/ランダムなエージェントの方が、私たちが工夫を重ねたヒューリスティックより
本番ラダーで高いスコアを記録している。** ローカルの`cg`シミュレータ上での対戦テスト
(FIXED_OPPONENTSプールへの勝率測定等)とは矛盾する結果であり、以下のいずれかが
起きている可能性が高い:

- Kaggle本番実行環境でのクラッシュ・タイムアウト(ローカルでは再現しない環境差異)
- 提出パッケージ(submission.tar.gz)の内容不備
- スコアリング方式(Glicko/Elo的レーティング?)の特性上、敗北を重ねるほど下がり、
  まだ評価されていない新規提出は中立的な初期値に留まる、等の仕組み的な要因

**→ 次にやるべき最優先タスク: 本番ラダーでの実際の敗因(クラッシュ/タイムアウト/弱い)を
特定する。** Kaggleの実行ログ・エラー出力が確認できるか調査し、可能なら再現する。
これが解決しない限り、ローカルでどれだけ強いエージェントを作っても本番スコアに
反映されない可能性がある。

## 2. 現状のメタ(独立分析: pilkwang "Meta Snapshot 06-29"より)

Kaggle Notebook `pilkwang/pok-mon-tcg-ai-battle-meta-snapshot-06-29` (2026-06-29時点、
高得票)の要旨:

- **Archaludon(メタルテンポ)が上昇中の最有力アーキタイプ**: 採用率上昇中かつ
  スコア率60%超。「Lucarioが使えるか」だけを問う段階はもう終わっている、との指摘。
- **Lucario系は頻出だが自己正当化できない**: よく見るが汎用的には弱く、操作精度・
  対面選択に強く依存する。
- **Starmie、Hop/Trevenantは依然「必須ストレステスト」**: 多くの一見快適な
  ファイティング系の対策がこの2つで崩れる。**現状の自分たちのFIXED_OPPONENTSプールには
  この2アーキタイプが無い**ため、今後の追加候補。
- **Alakazam/Dunsparceは有力な補完アーキタイプ**(Archaludonとは異なる失敗パターンを
  カバーする)。

→ 2026-06-30に追加した`alakazam_agent.py`/`archaludon_agent.py`は、この独立分析が
指摘する「今のメタで重要な2アーキタイプ」と一致している(Kaggle上位ノートブックの
人気・LBスコアの高さから選定したが、メタ分析の結論とも整合していたことを後から確認)。

未対応のメタ上重要アーキタイプ: **Starmie、Hop/Trevenant**(対戦相手プールへの
追加候補として記録)。

## 3. Discussion(フォーラム)について

KaggleのDiscussionタブはJSレンダリングのSPAで、CLI(`kaggle competitions`)にも
discussion関連サブコマンドが無いため直接の閲覧はできなかった。代わりに投票数上位の
Notebook群(`kaggle kernels list --competition pokemon-tcg-ai-battle --sort-by voteCount`)
が実質的に同種の戦略議論をカバーしている。今回参照したもの:

- `dashimaki360/beating-the-day-1-1-crustle-bot`(既知、Crustle対策の実戦記録)
- `pilkwang/pok-mon-tcg-ai-battle-meta-snapshot-06-29`(新規、上記2節の元ネタ)
- `smallpond/en-replay-archetype-analysis`(未調査、次回候補。実際のリプレイデータからの
  アーキタイプ分析)
- `kojimar/simple-baseline-matchup-tests`(未調査、対面別ベースラインテスト集)

## 4. 別コンペ「pokemon-tcg-ai-battle-challenge-strategy」について

調査中に発見。**$240,000 USDの賞金がついた別コンペ**(113チーム、締切2026-09-13)。
現在`userHasEntered=False`(未エントリー)。

- `kaggle competitions files`で確認した限り、提供ファイルはカードデータのみ
  (`cg`エンジン・sample_submission等の対戦シミュレータ一式が無い)。
- `kaggle competitions leaderboard`が空("No results found")。
- 関連Notebookには「Strategy Report」「EDA」「Analysis」系のタイトルが多い
  (例: `alphote/pka-strategy-report-pokemon-tcg-ai-battle`)。

→ おそらく**エージェント対戦ではなく、分析・戦略レポートNotebookを提出して審査される
コンペ**(賞金額からして本戦`pokemon-tcg-ai-battle`の関連企画/姉妹コンペの可能性)。
ルール詳細は未確認。賞金規模が大きいので、参加要否はユーザー判断で検討の余地あり
(本ドキュメントでは検出のみ行い、深追いはしていない)。

## 5. 今後のロードマップ(優先順)

1. **[最優先・新規]** 本番ラダーでの低スコアの原因調査(上記1節)。ローカルとの
   不整合を解消しないと他の改善が無意味になりかねない。
2. crustle/abomasnow対策込み継続学習(`mcts_loop_v4`、phase①)の評価結果を確認。
3. phase②: alakazam/archaludonも対戦相手プールに含めた追加学習。
4. 対戦相手プールにStarmie・Hop/Trevenantアーキタイプの追加を検討(本日の独立メタ分析より)。
5. (1の原因が判明し次第)現時点の最良モデル(ヒューリスティックまたはMCTS)を
   実際に本番提出し、本物のライブスコアで検証する。今回の session のRL/MCTS成果は
   まだ一度もKaggleに提出していない。
6. (任意・要ユーザー判断) `pokemon-tcg-ai-battle-challenge-strategy`への参加検討。
