# Handle both vector and matrix formats
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    s = np.array(x,float)
    if len(s.shape) > 1:
        for i in range( s.shape[1] ):
            sample = s[:,i]
            exp_sample = np.e**sample
            s[:,i] = exp_sample/exp_sample.sum()
        return s
    else:
        return np.e**s/(np.e**s).sum()


# One line version
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)



