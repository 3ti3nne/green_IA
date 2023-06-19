# Green IA

Tweaking a simple ML Model in Python to try and minimize time and resources while increasing performance and accuracy

## first tries

1. changed dtype in the model from float32 to 16, reduced precision but increased accuracy, memory gains and calculation speed

2. tried tweaking normalization of pixels by changing dividing number, changed accuracy but not a good way apparently

3. teaks in the Dense class, by tweaking the neurons number we can play with the capacity of the model to learn complex things more efficiently, but it can also cause an "overlearning" and reduce the accuracy

4. changed the optimizer, gained a lot of accuracy by changing adam to RMSprop, the optimizer is the algorithm that launches when we compile and train the model
