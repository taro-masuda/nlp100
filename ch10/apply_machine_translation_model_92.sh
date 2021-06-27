onmt_translate -model data/run/en_ja_model_step_1000.pt -src data/kftt-data-1.0/data/tok/kyoto-test.en -output data/run/pred_1000.txt -gpu 0 -verbose
# [2021-06-27 15:06:28,107 INFO] 
# SENT 1156: ['The', 'pronunciation', 'settled', 'on', "'", 'Daimyo', "'", 'after', 'the', 'beginning', 'of', 'the', 'Edo', 'period', ',', 'and', 'by', 'the', 'Kansei', 'era', 'they', 'were', 'solely', 'called', "'", 'Daimyo', '.', "'"]
# PRED 1156: <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> で あ る 。
# PRED SCORE: -40.8077