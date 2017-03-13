import sys
import urllib.request as ue

"""
Position of |'s, 0-based
24 (start)
71
171
271
343
"""


def read_alignments(target_url):
	def _get_nucs_from_reference(seq, ref):
		res_seq = ''
		for i in range(len(seq)):
			if seq[i] == '-':
				res_seq += ref[i]
			else:
				res_seq += seq[i]
			# if seq[i] == '-':
			# 	res_seq += ref[i]
			# elif seq[i] != '*' and seq[i] != '.':
			# 	res_seq += seq[i]
		return res_seq

	imgt_base = {}
	with ue.urlopen(target_url) as file:
		ref_flag = True
		reference = ''
		for line in file:
			line = line.decode().strip()
			if len(line) > 1:
				words = line.split()
				if len(words) > 1:
					words[1] = ''.join(words[1:])
					# if words[0][:len(gene) + 1] == gene + '*':
					if (words[0].find(':') != -1) and (words[0].find('*') != -1):  # ):
						if (ref_flag):
							# reference = words[1].replace('|', '')
							reference = words[1]
							ref_flag = False

						# imgt_base[words[0]] = imgt_base.get(words[0], '') + _get_nucs_from_reference(words[1].replace('|', ''), reference)
						imgt_base[words[0]] = imgt_base.get(words[0], '') + _get_nucs_from_reference(words[1], reference)
			else:
				ref_flag = True
	return imgt_base, max([len(x) for x in imgt_base.values()])


if __name__ == '__main__':
    # db = read_alignments(sys.argv[1])
    # db = read_alignments('https://raw.githubusercontent.com/jrob119/IMGTHLA/Latest/alignments/DQB_nuc.txt')
    db, max_len = read_alignments('https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/alignments/ClassI_prot.txt')
    stats = {"A": {}, "B": {}, "C": {}}
    with open("out_iclass.txt", 'w') as file:
        for x, val in sorted(db.items(), key = lambda x: x[0]):
            stats[x[0]][len(val)] = stats[x[0]].get(len(val), 0) + 1
            file.write(x + ' ' * (29 - len(x)) + val + "_" * (max_len - len(val)) + "\n")
    print(stats["A"], "\n")
    print(stats["B"], "\n")
    print(stats["C"], "\n")