import pickle
import numpy as np

def getDicFromPKL(path):
    with open(path, 'rb') as f:
        dic = pickle.load(f, encoding='bytes')
    return dic

def eval(final_scores):
    data_types = ['withSD', 'withoutSD', 'coscigan']
    for label in [0,1]:
        for data_type in data_types:
            scores = []
            for i in range(10):
                scores += [final_scores[f'iter_{i}_label_{label}'][data_type]]
            scores = np.array(scores)
            ave = scores.mean()
            std = scores.std()
            print('label: ', label, f';{data_type}: ', 'average %f, std %f' % (ave, std))


if __name__=='__main__':
    path = 'metric_discriminative/final_scores_withSD.pkl'
    dic = getDicFromPKL(path=path)
    eval(dic)
    
