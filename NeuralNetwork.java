import java.io.*;
import java.util.*;

public class NeuralNetwork
{
	/*
	 * Pointers for front and back of neural network
	 */
	private Node[] outputLayer;
	private Node[] inputLayer;

	/*
	 * Double arrays are linked to the double arrays in node Changing values in the
	 * double array will change value in node
	 */
	private List<double[]> weightData = new ArrayList<double[]>();
	private List<double[]> derWeightData = new ArrayList<double[]>();

	/*
	 * Rate of learning Momentum for moving average
	 */
	private double rate = 0.001;
	private double momentum = 1;

	/**
	 * Constructs a neural network with certain dimension and activation function
	 * 
	 * @param dim           Dimensions for network
	 * @param activation    Activation function for nodes
	 * @param activationDer Derivative of the activation function for nodes
	 */
	public NeuralNetwork(int[] dim, Activation activation, ActivationDer activationDer)
	{
		construct(dim, activation, activationDer);
	}

	/**
	 * Constructs a neural network with specification from file
	 * 
	 * @param fileName      Name of save file
	 * @param activation    Activation function for nodes
	 * @param activationDer Derivative of the activation function for nodes
	 */
	public NeuralNetwork(String fileName, Activation activation, ActivationDer activationDer)
	{
		File file = new File(fileName);
		try
		{
			/*
			 * Reads dimension data from first line To pass into construct
			 */
			Scanner scanner = new Scanner(file);

			String[] inString = scanner.nextLine().replace("[", "").replace("]", "").split(", ");
			int[] dim = new int[inString.length];

			for (int i = 0; i < inString.length; i++)
				dim[i] = Integer.parseInt(inString[i]);

			/*
			 * Constructs network with random weights To construct network and initialize
			 * weightData
			 */
			construct(dim, activation, activationDer);

			for (double[] weight : weightData)
			{
				inString = scanner.nextLine().replace("[", "").replace("]", "").split(", ");

				for (int i = 0; i < weight.length; i++)
					weight[i] = Double.parseDouble(inString[i]);
			}
		} catch (FileNotFoundException e)
		{
			System.out.print("File could not be found");
			return;
		} catch (Exception e)
		{
			System.out.print("File data error");
		}
	}

	/**
	 * Constructs layers for neural network
	 * 
	 * @param dim           Dimension of neural network
	 * @param activation    Activation function for nodes
	 * @param activationDer Derivative of the activation function for nodes
	 */
	private void construct(int[] dim, Activation activation, ActivationDer activationDer)
	{
		/*
		 * Initializes inputLayer and prevlayer To set up loop
		 */
		Node[] prevLayer = new Node[dim[0]];

		for (int i = 0; i < prevLayer.length; i++)
			prevLayer[i] = new Node(new Node[0], activation, activationDer);

		inputLayer = prevLayer;

		/*
		 * Creates a node layer of size dim[i] Initialize each node in layer with
		 * pointer prevLayer and activation function To link network
		 */
		for (int i = 1; i < dim.length; i++)
		{
			Node[] currLayer = new Node[dim[i]];

			for (int j = 0; j < dim[i]; j++)
				currLayer[j] = new Node(prevLayer, activation, activationDer);

			prevLayer = currLayer;
		}

		outputLayer = prevLayer;

		/*
		 * Run saveWeightData on each node in outputLayer To have a sorted list of all
		 * weights
		 */
		for (int i = 0; i < outputLayer.length; i++)
		{
			outputLayer[i].saveWeightData(i == 0);
			outputLayer[i].saveDerWeightData(i == 0);
		}
	}

	/**
	 * Calculates output of neural network
	 * 
	 * @param input Initial values for inputLayer
	 * @return Values of outputLayer in a double[]
	 */
	public double[] calc(double[] input)
	{
		for (int i = 0; i < inputLayer.length; i++)
			inputLayer[i].setVal(input[i]);

		double[] output = new double[outputLayer.length];

		for (int i = 0; i < output.length; i++)
			output[i] = outputLayer[i].getVal();

		clear();

		return output;
	}

	/**
	 * Clear stored value in nodes
	 */
	public void clear()
	{
		for (Node outputNode : outputLayer)
			outputNode.clear(outputLayer);
	}

	/**
	 * Clear stored gradient
	 */
	public void clearDer()
	{
		for (Node outputNode : outputLayer)
			outputNode.setDer(0);

		outputLayer[0].clearDer();
	}

	/**
	 * Calculates the derivatives of each node based on the error of prediction
	 * 
	 * @param input    Input for neural network
	 * @param expected The expected output from the neural network
	 * @return Total error of network output
	 */
	public double backProp(double[] input, double[] expected)
	{
		/*
		 * Initialize an error and sets the input layers value To generate output
		 */
		double error = 0;

		for (int i = 0; i < inputLayer.length; i++)
			inputLayer[i].setVal(input[i]);

		/*
		 * Set the output derivatives based on the derivatives of the error and adds to
		 * the total error To set up back propagation
		 */
		for (int i = 0; i < outputLayer.length; i++)
		{
			outputLayer[i].setDer(outputLayer[i].getVal() - expected[i]);
			error += 0.5 * Math.pow(outputLayer[i].getVal() - expected[i], 2);
		}

		/*
		 * Start recursion on each output node To calculate rest of node derivatives
		 */
		for (Node outputNode : outputLayer)
			outputNode.backProp(outputLayer);

		clear();

		return error;
	}

	/**
	 * Calculates the derivatives of each node based on the given gradients
	 * 
	 * @param input     Inputs for neural network
	 * @param gradients Gradients for output layer
	 */
	public void backPropGrad(double[] input, double[] gradients)
	{
		/*
		 * Set value for the input layer
		 */
		for (int i = 0; i < inputLayer.length; i++)
			inputLayer[i].setVal(input[i]);
		/*
		 * Set gradient for output layer
		 */
		for (int i = 0; i < outputLayer.length; i++)
			outputLayer[i].setDer(gradients[i]);

		/*
		 * Start recursion on each output node To calculate rest of node derivatives
		 */
		for (Node outputNode : outputLayer)
			outputNode.backProp(outputLayer);

		clear();
	}

	/**
	 * Updates weights based on gradients and clear stored gradients
	 */
	public void updateWeight()
	{
		for (Node outputNode : outputLayer)
			outputNode.updateWeight();
	}
	
	/**
	 * Sets learning rate -Decrease rate each update for rate decay
	 * 
	 * @param rate New rate
	 */
	public void setRate(double rate)
	{
		this.rate = rate;
	}

	/**
	 * Exponential moving average, A higher momentum discounts older gradients
	 * faster. Momentum has values of (0,1]
	 * 
	 * @param momentum Amount of gradients that remains after update
	 */
	public void setMomentum(double momentum)
	{
		this.momentum = momentum;
	}
	
	/**
	 * Set a different activation function for the output layer
	 * 
	 * @param activate    Activation function for output layer
	 * @param activateDer Derivative of activation function for output layer
	 */
	public void setOutputActivation(Activation activate, ActivationDer activateDer)
	{
		for (Node outputNode : outputLayer)
			outputNode.setActivation(activate, activateDer);
	}
	
	/**
	 * @return	Weights in a list
	 */
	public List<double[]> getWeightData()
	{
		return weightData;
	}
	
	/**
	 * @return	derWeights in a list
	 */
	public List<double[]> getDerWeightData()
	{
		return derWeightData;
	}

	/**
	 * 
	 * @return Memory location of first layer
	 */
	public Node[] getInputLayer()
	{
		return inputLayer;
	}

	/**
	 * 
	 * @return Memory location of last layer
	 */
	public Node[] getOutputLayer()
	{
		return outputLayer;
	}

	/**
	 * Calculates the gradient of the input layer
	 * 
	 * @param input    Input for neural network
	 * @param expected Expected output for neural network
	 * @return Double array containing gradient of the input layer
	 */
	public double[] getInputDer(double[] input, double[] expected)
	{
		/*
		 * Set value for input layer
		 */
		for (int i = 0; i < inputLayer.length; i++)// initialize networkBack
			inputLayer[i].setVal(input[i]);

		/*
		 * Set output gradient based on derivative of error
		 */
		for (int i = 0; i < outputLayer.length; i++)
			outputLayer[i].setDer(outputLayer[i].getVal() - expected[i]);// sets derivative for output layer

		/*
		 * Start recursion on each output node To calculate rest of node derivatives
		 */
		for (Node outputNode : outputLayer)
			outputNode.backProp(outputLayer);

		/*
		 * Create output array
		 */
		double[] layerDer = new double[inputLayer.length];

		for (int i = 0; i < layerDer.length; i++)
			layerDer[i] = inputLayer[i].getDer();

		clear();

		return layerDer;
	}

	public int[] getDim()
	{
		int length = 1;

		for (Node[] curLayer = outputLayer; !curLayer.equals(inputLayer); curLayer = curLayer[0].getPrevLayer())
			length++;

		Node[] curLayer = outputLayer;
		int[] dim = new int[length];

		for (length -= 1; length >= 0; length--)
		{
			dim[length] = curLayer.length;
			curLayer = curLayer[0].getPrevLayer();
		}

		return dim;
	}

	/**
	 * Saves the current state of the neural network to a file so it may be used in
	 * another instance
	 * 
	 * @param fileName Location of file on system
	 */
	public void saveNetwork(String fileName)
	{
		/*
		 * Generates array which contains dimension of neural network
		 */
		int[] dim = getDim();

		/*
		 * Writes array of weights for each node. Each node's weights are on a line
		 */
		try
		{
			File file = new File(fileName);
			file.createNewFile();

			FileWriter fileWriter = new FileWriter(file.getAbsolutePath());

			fileWriter.write(Arrays.toString(dim) + "\n");

			for (double[] w : weightData)
				fileWriter.write(Arrays.toString(w) + "\n");

			fileWriter.flush();
			fileWriter.close();
		} catch (Exception e)
		{
			System.out.print("network could not be saved");
		}
	}

	/**
	 * Numerically check result for back propagate, use for debug
	 * 
	 * @param input    Input for neural network
	 * @param expected The expected output from the neural network
	 */
	public void testDer(double[] input, double[] expected)
	{
		/*
		 * Get a the derivatives of each weight
		 */
		List<double[]> approxDerWeightData = new ArrayList<double[]>();

		double initialError = backProp(input, expected);

		for (int i = 0; i < outputLayer.length; i++)
			outputLayer[i].saveDerWeightData(i == 0);

		for (int i = 0; i < weightData.size(); i++)
		{
			approxDerWeightData.add(new double[weightData.get(i).length]);

			for (int j = 0; j < weightData.get(i).length; j++)
			{
				double error = 0;
				weightData.get(i)[j] += 0.000001;

				double[] output = calc(input);
				for (int k = 0; k < outputLayer.length; k++)
					error += 0.5 * Math.pow(output[k] - expected[k], 2);
				approxDerWeightData.get(i)[j] = (error - initialError) / 0.000001;

				weightData.get(i)[j] -= 0.000001;
				clear();
			}
		}
		clear();
	}

	public class Node
	{
		private Node[] prevLayer;

		private double val;
		private boolean isVal;
		private double preAct;

		private double der;
		private boolean isDer;

		private double[] weight;
		private double[] derWeight;

		private Activation activation;
		private ActivationDer activationDer;

		/**
		 * Creates a node with weights connecting to the layer behind it
		 * 
		 * @param inLayer
		 * @param activate    Activation function
		 * @param activateDer Derivative of the activation function
		 */
		public Node(Node[] inLayer, Activation activate, ActivationDer activateDer) // and set random weights
		{
			prevLayer = inLayer;
			weight = new double[prevLayer.length + 1];
			derWeight = new double[prevLayer.length + 1];

			for (int i = 0; i < weight.length; i++)
				weight[i] = (Math.random() * 2 - 1) * 0.1;

			activation = activate;
			activationDer = activateDer;
		}

		/**
		 * If node value has been calculated, current value. Otherwise calculate it.
		 * 
		 * @return Value of node
		 */
		public double getVal()
		{
			if (isVal)
				return val;
			/*
			 * preAct is the weighted sum of prevLayer initialize preAct as bias value
			 */
			preAct = weight[weight.length - 1];

			for (int i = 0; i < prevLayer.length; i++)
				preAct += prevLayer[i].getVal() * weight[i];

			val = activation.activate(preAct);

			isVal = true;

			return val;
		}

		/**
		 * @return Value of node before activation function
		 */
		public double getPreAct()
		{
			return preAct;
		}

		/**
		 * @param input Value of current node
		 */
		public void setVal(double input)
		{
			val = input;
			isVal = true;
		}

		/**
		 * @return Memory address of previous layer
		 */
		public Node[] getPrevLayer()
		{
			return prevLayer;
		}

		/**
		 * 
		 * @return Current stored derivative
		 */
		public double getDer()
		{
			return der;
		}

		/**
		 * Sets derivative of node and calculates derivative of weights Assumes input
		 * layer has been initialized
		 * 
		 * @param der Derivative of the current node
		 */
		public void setDer(double der)
		{
			this.der = der;

			/*
			 * Adds current weight derivative to the stored weight derivatives
			 */
			for (int i = 0; i < prevLayer.length; i++)
				derWeight[i] += momentum * prevLayer[i].getVal() * activationDer.activateDer(val, preAct) * der;
			/*
			 * Last weight in array corresponds to bias
			 */
			derWeight[derWeight.length - 1] += momentum * activationDer.activateDer(val, preAct) * der;
			isDer = true;
		}

		/**
		 * @return Memory address for the array of weights
		 */
		public double[] getWeight()
		{
			return weight;
		}

		/**
		 * @return Memory address for the array of stored derivatives
		 */
		public double[] getDerWeight()
		{
			return derWeight;
		}

		/**
		 * @param activate    Activation function for node
		 * @param activateDer derivative activation function for node
		 */
		public void setActivation(Activation activate, ActivationDer activateDer)
		{
			activation = activate;
			activationDer = activateDer;
		}

		/**
		 * Adds the current weight array to the list weightData
		 * 
		 * @param weightData Place for storage of current weight array
		 * @param recur      Whether or not to recur to all nodes in prevLayer
		 * @param inputLayer Node layer for which to stop recursion
		 */
		public void saveWeightData(boolean recur)
		{
			/*
			 * Do recursion if current node is at index 0 and its not inputLayer
			 */
			if (recur && !prevLayer.equals(inputLayer))
			{
				for (int i = 0; i < prevLayer.length; i++)
					prevLayer[i].saveWeightData(i == 0); // saveWeightData will recur to all nodes in previous layer if
															// argument is true
			}

			weightData.add(weight);
		}

		/**
		 * Adds the current weight gradient array to the list derWeightData
		 * 
		 * @param weightData Place for storage of current weight gradient array
		 * @param recur      Whether or not to recur to all nodes in prevLayer
		 * @param inputLayer Node layer for which to stop recursion
		 */
		public void saveDerWeightData(boolean recur)
		{
			/*
			 * Do recursion if current node is at index 0 and its not inputLayer
			 */
			if (recur && !prevLayer.equals(inputLayer))
			{
				for (int i = 0; i < prevLayer.length; i++)
					prevLayer[i].saveDerWeightData(i == 0);
			}

			derWeightData.add(derWeight);
		}

		/**
		 * Calculates the derivatives for the nodes in previous layer. Assumes gradient
		 * for weights in current layer is calculated
		 * 
		 * @param currLayer Array with all the nodes in the same layer as current node
		 */
		public void backProp(Node[] currLayer)
		{
			/*
			 * Calculate the gradients for the nodes in previous layer by summation of
			 * derivatives of weights
			 */
			for (int i = 0; i < prevLayer.length; i++)
			{
				double sumDer = 0;

				for (Node curNode : currLayer)
					sumDer += activationDer.activateDer(curNode.getVal(), curNode.getPreAct()) * curNode.getWeight()[i]
							* curNode.getDer();

				prevLayer[i].setDer(sumDer);
			}

			/*
			 * Recursive call unless previous layer is input layer
			 */
			if (!prevLayer.equals(inputLayer))
				prevLayer[0].backProp(prevLayer);
		}

		/**
		 * Update weights and clear the current stores gradient
		 */
		public void updateWeight()
		{
			/*
			 * Return if derivative is cleared
			 */
			if (isDer == false)
				return;

			/*
			 * Update weight by a normalized gradient. A portion of the gradient is not
			 * removed, amount that remains is determined by the momentum
			 */
			for (int i = 0; i < weight.length - 1; i++)
			{
				weight[i] -= clip(derWeight[i] * rate);
				prevLayer[i].updateWeight();
				derWeight[i] *= 1.0 - momentum;
			}

			/*
			 * Last weight in array is bias
			 */
			weight[weight.length - 1] -= clip(derWeight[derWeight.length - 1] * rate);
			derWeight[derWeight.length - 1] *= 1.0 - momentum;
			isDer = false;
		}

		/**
		 * Clips derivative if values are large
		 * 
		 * @param der Input derivative
		 * @return Clipped derivative
		 */
		private double clip(double der)
		{
			if (der < 0.05 && der > -0.05)
				return der;
			if (der > 0)
				return 0.05;
			return -0.05;
		}

		/**
		 * Clears value in node and recursively clears previous layer
		 * 
		 * @param currLayer
		 */
		public void clear(Node[] currLayer)
		{
			if (!isVal)
				return;

			val = 0;
			isVal = false;

			if (!currLayer.equals(inputLayer))
			{
				for (int i = 0; i < prevLayer.length; i++)
					prevLayer[i].clear(prevLayer);
			}
		}

		/**
		 * Reset stored derivatives to 0
		 */
		public void clearDer()
		{
			for (Node prevNode : prevLayer)
				prevNode.setDer(0.0);

			if (!prevLayer.equals(inputLayer))
				prevLayer[0].clearDer();
		}
	}

	interface Activation
	{
		double activate(double val);
	}

	interface ActivationDer
	{
		double activateDer(double val, double preAct);
	}
}