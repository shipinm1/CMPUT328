Each learning function takes three parameters: (x_test,mnist,y_test)

The only thing change in the main.py file is: 

predicted_y_test = algorithm.run(x_test)
to
predicted_y_test = algorithm.run(x_test,mnist,y_test)