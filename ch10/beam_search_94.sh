for i in `seq 100`
do
    onmt_translate -beam_size $i -model data/run/en_ja_model_step_1000.pt -src data/kftt-data-1.0/data/tok/kyoto-test.en -output data/run/pred_1000_beam_$i.txt -gpu 0 -verbose
done

python ch10/beam_search_94.py 