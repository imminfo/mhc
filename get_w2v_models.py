import numpy as np
from mhystic.embedding import *
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if "uniprot.fasta" in os.listdir("./data/"):
    pass
else:
    print("No Uniprot data. Run : wget http://www.uniprot.org/uniprot/?query=human&format=fasta&compress=yes")
    quit()

def get_9mers(seqs):
    mers9 = []
    for cur_seq in seqs:
        cur_len = len(cur_seq) - len(cur_seq)%9
        for i in range(0, cur_len, 9):
            if re.search('[X|Z|B|O|U]', cur_seq[i:i+9]) != None:
                print("FOUND", cur_seq[i:i+9])
                continue
            mers9.append(cur_seq[i:i+9])
    return list(pd.Series(mers9).apply(fill_spaces))

big_sequence = []
for seq_record in SeqIO.parse("./data/uniprot.fasta", "fasta"):
    big_sequence.append(str(seq_record.seq))

w2v_sequences = get_9mers(big_sequence)
# print(w2v_sequences)

if "9mers.npy" in os.listdir("./data/"):
    pass
else:
    np.save("./data/9mers.npy", np.array(w2v_sequences))

for i in [10, 20, 50, 80]:
    w2v_model = train_seq2vec(w2v_sequences, w2v_dim=i, num_workers=60)
    print("\n" + 30*"=" + "\n"+ "TRAINING W2V FOR SIZE {} DONE\n".format(i)+30*"="+"\n")
    w2v_model.save("./w2v_models/up9mers_size_{}_window_3.pkl".format(i))
