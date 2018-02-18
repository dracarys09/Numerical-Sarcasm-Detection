i = 0
with open('data.tsv', 'a+') as f:
    f.write('id\tsentiment\treview\n')
    with open('sarcastic_pos.txt') as pos:
        sarcastic_pos = pos.readlines()
        for line in sarcastic_pos:
            f.write('"'+str(i)+'"'+'\t1'+'\t'+'"'+line.strip()+'"\n')
            i = i+1
    with open('nonsarcastic_neg.txt') as neg:
        nonsarcastic_neg = neg.readlines()
        for line in nonsarcastic_neg:
            f.write('"'+str(i)+'"'+'\t0'+'\t'+'"'+line.strip()+'"\n')
            i = i+1
