with open('./token_attentions', 'r') as f:
    with open('./low_rated_word', 'w') as f2:
        for i in range(0, 5000):
            a = f.readline()
            f2.write(a.split('  ||  ')[0] + '\n')
