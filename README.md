# Self_Organizing_Map_Feature_Detection_With_ANN
A self organizing map that uses customer data to identify fraudulent customers. Then, the entire data is used to train a vanilla neural network to identify fraudulent behaviour.

Data is obtained containing customers that applied for a credit card. Customers are represented with a customer ID, 14 other features, and finally a class indicating wether or not they have been approved yet for a credit card. The data is preprocessed and is used to train a 10x10 self organizing map. Neurons that are furthest away from other neurons represent outliers and therefore fraudulent behabiour. All customers that have any of those neurons as their winning neuron are labelled as fradulent.

Using our new dataset, which includes everything as before, with an additional class of if the customer is fradulent or not fradulent, a vanilla neural netword is trained to learn the characteristics that make a customer display fradulent or nonfraudulent behaviour.
