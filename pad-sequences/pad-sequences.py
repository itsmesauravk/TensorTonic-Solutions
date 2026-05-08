import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len == None:
        max_len = 0
        for seq in seqs:    
            seq_len = len(seq)
            if seq_len > max_len:
                max_len = seq_len

    #pad and truncat
    padded_seqs = []
    for seq in seqs:
        if len(seq) == max_len:
            padded_seqs.append(seq)
            
        elif len(seq) < max_len:
            num_pad = max_len - len(seq) 
            pad_seq = np.pad(seq, (0, num_pad), mode='constant', constant_values=pad_value)
            padded_seqs.append(pad_seq)

        else:
            pad_seq = seq[:max_len]
            padded_seqs.append(pad_seq)

    return padded_seqs
            
        
    
            
    