import json

from decomp import UDSCorpus

uds = UDSCorpus()

with open('test_4.txt', 'w') as f:

    print(uds["ewt-train-11161"].semantics_edges('ewt-train-11161-semantics-pred-3%%ewt-train-11161-semantics-arg-2'))
    print(uds["ewt-train-12"].head('ewt-train-12-semantics-pred-7', ['form','lemma']))

#    #print(uds["ewt-train-12"].nodes)
#    for node in uds["ewt-train-11161"].nodes:
#        print(uds["ewt-train-11161"].nodes[node])
#        cur_dict = uds["ewt-train-11161"].nodes[node]
#        #print(cur_dict['form'])

 



