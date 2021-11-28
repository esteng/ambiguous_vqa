import json

from decomp import UDSCorpus

uds = UDSCorpus()

with open('test_4.txt', 'w') as f:
    #print(uds["ewt-train-12"].nodes)
    for node in uds["ewt-train-11161"].nodes:
        print(uds["ewt-train-11161"].nodes[node])
        cur_dict = uds["ewt-train-11161"].nodes[node]
        #print(cur_dict['form'])

#    for graph in uds:
#        for node in graph.nodes:
#            print(graph.nodes[node])

#print(uds["ewt-train-12"])

#print(uds["ewt-train-11161"].sentence)
#dict = uds["ewt-train-11161"].semantics_nodes
#with open('test_2.txt', 'w') as f:
    #print(uds["ewt-train-2550"]))

#for graphid, graphs in uds.items():
#    print(graphid)
#    print(graph.sentence)

#querystr = """
#            SELECT ?pred
#            WHERE { ?pred <domain> <semantics> ; <type> <predicate> ; <factual> ?factual ; <dur-minutes> ?duration FILTER ( ?factual > 0 && ?duration > 0 )
#            }
#            """
#with open('test_2.txt', 'w') as f:
    



