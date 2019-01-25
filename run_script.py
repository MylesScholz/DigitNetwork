#Run Script
#Myles Scholz

import mnist_loader as dp
import Network as nw

tr_d, te_d, va_d = dp.load_data_wrapper()
print(str(len(tr_d)) + ", " + str(len(te_d)))
net = nw.Network([784,100,30,11])
net.SGD(tr_d,30,10,3.0,test_data=te_d)
if input("Save network? [y/n]:") == "y": net.save_network("num_net3.p")