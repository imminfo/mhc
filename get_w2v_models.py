import numpy as np
from mhystic.embedding import *
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import re

if "uniprot.fasta" in os.listdir("./data/"):
    pass
else:
    print("No Uniprot data. Run : wget http://www.uniprot.org/uniprot/?query=human&format=fasta&compress=yes")
    quit()

def get_9mers(seqs):
    mers9 = []
    for cur_seq in seqs:
        cur_len = len(cur_seq) - len(cur_seq)%9
        for i in range(0, cur_len):
            if re.search('[X|Z|B|O|U]', cur_seq[i:i+9]):
                # print("FOUND", cur_seq[i:i+9])
                continue
            mers9.append(cur_seq[i:i+9])
    return list(pd.Series(mers9).apply(fill_spaces))


big_sequence = [str(seq_record.seq) for seq_record in SeqIO.parse("./data/uniprot.fasta", "fasta")]
# for seq_record in SeqIO.parse("./data/uniprot.fasta", "fasta"):
#     big_sequence.append(str(seq_record.seq))

w2v_sequences = get_9mers(big_sequence)
# print(w2v_sequences)


# if "9mers.npy" in os.listdir("./data/"):
#     pass
# else:
#     np.save("./data/9mers.npy", np.array(w2v_sequences))

# for i in [10, 20, 50, 80]:
for i in [20, 50, 80]:
    w2v_model = train_seq2vec(w2v_sequences, w2v_dim=i, num_workers=30)
    print("\n" + 30*"=" + "\n"+ "TRAINING W2V FOR SIZE {} DONE\n".format(i)+30*"="+"\n")
    w2v_model.save("./w2v_models/up9mers_size_{}_window_3.pkl".format(i))


# if "uniprot.fasta" in os.listdir("./data/"):
#     pass
# else:
#     print("No Uniprot data. Run : wget http://www.uniprot.org/uniprot/?query=human&format=fasta&compress=yes")
#     quit()

# def get_w2v_model(path_to_fasta, dim):
#     a=0
#     mers9 = []
#     for seq_record in SeqIO.parse(path_to_fasta, "fasta"):
#         a+=1
#         if a == 5:
#             break
#         cur_seq = str(seq_record.seq)
#         cur_len = len(cur_seq) - len(cur_seq)%9
#         for i in range(0, cur_len, 1):
#             if re.search('[X|Z|B|O|U]', cur_seq[i:i+9]) != None:
#                 continue
#             mers9.append(cur_seq[i:i+9])

#     mers9 = list(pd.Series(mers9).apply(fill_spaces))

#     w2v_model = train_seq2vec(mers9, w2v_dim=dim, num_workers=60)

#     return w2v_model

# for i in [10, 20, 50, 80]:
#     w2v_model_dim = get_w2v_model("./data/uniprot.fasta", i)
#     print("\n" + 30*"=" + "\n"+ "TRAINING W2V FOR SIZE {} DONE\n".format(i)+30*"="+"\n")
#     w2v_model_dim.save("./w2v_models/up9mers_size_{}.pkl".format(i))

