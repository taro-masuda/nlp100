from nltk.translate.bleu_score import corpus_bleu
from bleu_score_93 import append_lines
import matplotlib.pyplot as plt

if __name__ == "__main__":
    refpath = 'data/kftt-data-1.0/data/tok/kyoto-test.ja'
    figpath = 'data/beamsearch.png'

    ref = append_lines(path=refpath, mode='ref')

    plt.xlabel('beam length')
    plt.ylabel('BLEU score')

    for beam_len in range(1, 100+1):
        hypopath = 'data/run/pred_1500_beam_' + str(beam_len) + '.txt'
        hypo = append_lines(path=hypopath, mode='hypo')
        score = corpus_bleu(ref, hypo)
        print('Corpus BLEU score:', score)
        plt.scatter(beam_len, score, color='b')
        
    plt.savefig(figpath)