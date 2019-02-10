neural = Network([6,5,3])
neural.SGD([[[0,1,1,0,1,0], [0,1,0]]],3)
neural.feed_network([0,1,1,0,1,0])