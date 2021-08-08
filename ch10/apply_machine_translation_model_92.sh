onmt_translate -model data/run/en_ja_model_step_5000.pt -src data/kftt-data-1.0/data/tok/kyoto-test.en -output data/run/pred_5000.txt -gpu 0 -verbose
# [2021-06-27 15:06:28,107 INFO] 
# SENT 1156: ['The', 'pronunciation', 'settled', 'on', "'", 'Daimyo', "'", 'after', 'the', 'beginning', 'of', 'the', 'Edo', 'period', ',', 'and', 'by', 'the', 'Kansei', 'era', 'they', 'were', 'solely', 'called', "'", 'Daimyo', '.', "'"]
# PRED 1156: <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> の <unk> で あ る 。
# PRED SCORE: -40.8077
# 
# [2021-08-07 16:26:47,182 INFO] PRED AVG SCORE: -1.0332, PRED PPL: 2.8099