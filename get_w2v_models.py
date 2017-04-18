import numpy as np
from mhystic.embedding import *
import re
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if "uniprot.fasta" in os.listdir("./data/"):
    pass
else:
    print("No Uniprot data. Run : wget http://www.uniprot.org/uniprot/?query=human&format=fasta&compress=yes")
    quit()

def get_w2v_model(path_to_fasta, dim):
    a=0
    mers9 = []
    for seq_record in SeqIO.parse(path_to_fasta, "fasta"):
        a+=1
        if a == 5:
            break
        cur_seq = str(seq_record.seq)
        cur_len = len(cur_seq) - len(cur_seq)%9
        for i in range(0, cur_len, 1):
            if re.search('[X|Z|B|O|U]', cur_seq[i:i+9]) != None:
                continue
            mers9.append(cur_seq[i:i+9])

    mers9 = list(pd.Series(mers9).apply(fill_spaces))

    w2v_model = train_seq2vec(mers9, w2v_dim=dim, num_workers=60)

    return w2v_model

for i in [10, 20, 50, 80]:
    w2v_model_dim = get_w2v_model("./data/uniprot.fasta", i)
    print("\n" + 30*"=" + "\n"+ "TRAINING W2V FOR SIZE {} DONE\n".format(i)+30*"="+"\n")
    w2v_model_dim.save("./w2v_models/up9mers_size_{}.pkl".format(i))
