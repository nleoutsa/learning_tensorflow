MAX_LEN = 100
f = open("datasets/friends_transcripts/friends_transcripts_eighth.txt")
text = f.read()
lines = text.lower().split("\n")

# truncate lines to help with OOM issues
truncated = []
for x in lines:
    if len(x) > MAX_LEN:
        truncated.append(' '.join(x[0:MAX_LEN].split()[:-1]))
    else:
        truncated.append(x)

corpus = truncated
