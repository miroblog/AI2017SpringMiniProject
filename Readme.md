# Mini Project for Spring17 A.I. Course

### Experiment with MLP
* `# Layers, # Neurons`
* `L2 normalization`
* `Dropout`
* `BatchNormalization`
* `Objective Functions`
* `Activation Functions`

[Report]()
[Code]
* [MNIST] mnist_architecture_tuning.py, mnist_objective_func.py
* [Cifar10] cifar10_architecture_tuning.py, cifar10_objective_func.py   

## Getting Started

```python
	$ python3 mnist_architecture_tuning.py
	$ python3 mnist_objective_func.py
	$ python3 cifar10_architecture_tuning.py 
	$ python3 cifar10_objective_func.py

	mnist/cifar10 _architecture_tuning.py :  
	* hidden layer, unit 
	* learning rate
	* activation function
	* dropout - rate
	* batch normalisation
 	등을 조정할 수 있다.	
	
	mnist/cifar10_objective
	* loss function : cross entropy, L2 norm, mean squared error
	를 조정할 수 있다.

	(e.g.)
		for n_hidden_layer in [1, 2, 3]: 
			for n_hidden_unit in [10, 50, 100, 200, 400, 800]:
			...
	실험하고자 하는 설정을 list에다가 채워넣으면 된다.

```

### Prerequisites

tensorflow, sklearn, nuumpy, pandas ...

```python
pip install -r requirements.txt
```

## Authors

* **Lee Hankyol** - *Initial work* - [L2-Prediction](https://github.com/miroblog/limit_orderbook_prediction)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
