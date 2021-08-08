from nltk.translate.bleu_score import corpus_bleu

def append_lines(path: str, mode: str) -> list:
    if not (mode == 'hypo' or mode == 'ref'):
        raise ValueError
    lists = []
    with open(file=path, mode='r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.split()
            if mode == 'hypo':
                lists.append(words)
            elif mode == 'ref':
                lists.append([words])
    return lists

if __name__ == "__main__":

    hypopath = 'data/run/pred_5000.txt'
    refpath = 'data/kftt-data-1.0/data/tok/kyoto-test.ja'

    hypo = append_lines(path=hypopath, mode='hypo')
    ref = append_lines(path=refpath, mode='ref')

    print('Corpus BLEU score:', corpus_bleu(ref, hypo))
    # Corpus BLEU score: 0.121311235142337