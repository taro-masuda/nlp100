for i in `seq 1 100`
do
    onmt_translate -beam_size $i -model data/run/en_ja_model_step_1500.pt -src data/kftt-data-1.0/data/tok/kyoto-test.en -output data/run/pred_1500_beam_$i.txt -gpu 0 -verbose
done