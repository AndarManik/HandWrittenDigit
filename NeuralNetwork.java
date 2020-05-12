import java.io.*;
import java.util.*;

public class NeuralNetwork
{
   private int[] dim;
   
   private Node[] outputLayer;
   private Node[] inputLayer;
   
   private List<double[]> weightData = new ArrayList<double[]>();
   private List<double[]> derWeightData = new ArrayList<double[]>();
   
   private double rate = 0.001;
   private double momentum = 1;
   
   private Activation activation = (val) -> (Math.exp(val) - Math.exp(val * -1)) / (Math.exp(val) + Math.exp(val * -1));
   private ActivationDer activationDer = (val, preAct) -> 1 - val * val;
   
   public NeuralNetwork(int[] dim, Activation activate, ActivationDer activateDer)//creates networkBack with dimensions, and with different activation;
   {
      this.dim = dim;
      activation = activate;
      activationDer = activateDer;
      
      construct();
   }
   
   public NeuralNetwork(String fileName, Activation activate, ActivationDer activateDer) throws FileNotFoundException //creates a NeuralNetwork with activation function which is saved in file
   {
		   activation = activate;
		   activationDer = activateDer;
		      
		   File file = new File(fileName);
		   Scanner scanner = new Scanner(file);
		   
		   String[] inString = scanner.nextLine().replace("[", "").replace("]", "").split(", ");
		   dim = new int[inString.length];
		   
		   for(int i = 0; i < inString.length; i++)
			   dim[i] = Integer.parseInt(inString[i]);
		   
		   construct();
		   
		   for(double[] weight: weightData)
		   {
			   inString = scanner.nextLine().replace("[", "").replace("]", "").split(", ");
			   
			   for(int i = 0; i < weight.length; i++)
				   weight[i] = Double.parseDouble(inString[i]);
		   }
   }
   
   public void construct()
   {
	   	Node[] prevLayer = new Node[dim[0]];//create input layer node array using nodeFront as the prevLayer
	      
	    for(int i = 0; i < prevLayer.length; i++)
	    	prevLayer[i] = new Node(new Node[0], activation, activationDer);
	    
	    inputLayer = prevLayer;
	         
	    for(int i = 1; i < dim.length; i++)//loop through all the layers
	    {
	    	Node[] currLayer = new Node[dim[i]];//create hidden layer 
	         
	    	for(int j = 0; j < dim[i]; j++)
	    		currLayer[j] = new Node(prevLayer, activation, activationDer);//link previous layer using 1 arg constructor
	            
	        prevLayer = currLayer;
	    }
	      
	    outputLayer = prevLayer;
	      
		for(int i = 0; i < outputLayer.length; i++)
		{
			outputLayer[i].saveWeightData(weightData, i == 0, inputLayer);
			outputLayer[i].saveDerWeightData(derWeightData, i == 0, inputLayer);
		}
   }
   
   public double[] calc(double[] input)// forward pass of network
   {  
      for(int i = 0; i < inputLayer.length; i++)//initialize networkBack
         inputLayer[i].setVal(input[i]);
      
      double[] output = new double[outputLayer.length];
      
      for(int i = 0; i < output.length; i++)
         output[i] = outputLayer[i].getVal();//set values for output
      
      clear();
      
      return output;
   }
   
   public void clear()
   {
	   for(Node outputNode: outputLayer)
		   outputNode.clear(outputLayer);
   }
   
   public void clearDer()//clear the stored gradient
   {
	   for(Node outputNode: outputLayer)
		   outputNode.setDer(0);
	   
	   outputLayer[0].clearDer();
   }
   
   public double backProp(double[] input, double[] expected)// calculates and add the current cost gradient to the stored cost gradient: calculation based on expectation
   {
	  double error = 0;
	  
	  for(int i = 0; i < inputLayer.length; i++)//initialize networkBack
	         inputLayer[i].setVal(input[i]);
      
      for(int i = 0; i < outputLayer.length; i++)
      {
         outputLayer[i].setDer(outputLayer[i].getVal() - expected[i]);//sets derivative for output layer 
         error += 0.5 * Math.pow(outputLayer[i].getVal() - expected[i], 2);
      }
      
      for(Node outputNode: outputLayer)
		   outputNode.backProp(outputLayer);
      
      clear();
      
      return error;
   }
   
   public void backPropGrad(double[] input, double[] gradients)// backProp against a calculated gradient
   {
	   for(int i = 0; i < inputLayer.length; i++)//initialize networkBack
		  inputLayer[i].setVal(input[i]);
	      
	   for(int i = 0; i < outputLayer.length; i++)
		  outputLayer[i].setDer(gradients[i]);//sets derivative for output layer 
	      
	   for(Node outputNode: outputLayer)
		   outputNode.backProp(outputLayer);
	      
	   clear();
   }
   
   public void updateWeight()//update weights based on gradients and clear stored gradients
   {
      for(Node outputNode: outputLayer)
         outputNode.updateWeight();
   }
   
   public void setRate(double rate)
   {
	   this.rate = rate;
   }
   
   public void setMomentum(double momentum)
   {
	   this.momentum = momentum;
   }
   
   public int[] getDim()
   {
	   return dim;
   }
   
   public Node[] getInputLayer()
   {
	   return inputLayer;
   }
   
   public Node[] getOutputLayer()
   {
	   return outputLayer;
   }
   
   public double[] getInputDer(double[] input, double[] expected)
   {   
	   for(int i = 0; i < inputLayer.length; i++)//initialize networkBack
		         inputLayer[i].setVal(input[i]);
	      
	   for(int i = 0; i < outputLayer.length; i++)
	      outputLayer[i].setDer(outputLayer[i].getVal() - expected[i]);//sets derivative for output layer 
	      
	      for(Node outputNode: outputLayer)
			   outputNode.backProp(outputLayer);
	      
	   double[] layerDer = new double[inputLayer.length];
		   
	   for(int i = 0; i < layerDer.length; i++)
		   layerDer[i] = inputLayer[i].getDer();
	   
	   clear();
	      
	   return layerDer;
   }
   
   public void outputActivation(Activation activate, ActivationDer activateDer)
   {
	   for(Node outputNode: outputLayer)
		   outputNode.setActivation(activate, activateDer);
   }
   
   public String saveNetwork(String fileName)
   {
	   try
	   {
		   File file = new File(fileName);
		   file.createNewFile();
		   
		   FileWriter fileWriter = new FileWriter(file.getAbsolutePath());
		   
		   fileWriter.write(Arrays.toString(dim) + "\n");
		   
		   for(double[] w: weightData)
			   fileWriter.write(Arrays.toString(w) + "\n");
		   
		   fileWriter.flush();
		   fileWriter.close();
		   
		   return file.getAbsolutePath();
	   }
	   catch(Exception e)
	   {
		   return e.toString();
	   }
   }
   
   public void testDer(double[] input, double[] expected)
   {
	   List<double[]> derWeightData = new ArrayList<double[]>();
	   List<double[]> approxDerWeightData = new ArrayList<double[]>();
	   
	   double initialError = backProp(input, expected);
	   
	   for(int i = 0; i < outputLayer.length; i++)
			outputLayer[i].saveDerWeightData(derWeightData, i == 0, inputLayer);
	   
	   for(int i = 0; i < weightData.size(); i++)
	   {
		   approxDerWeightData.add(new double[weightData.get(i).length]);
		   
		   for(int j = 0; j < weightData.get(i).length; j++)
		   {
			   double error = 0;
			   weightData.get(i)[j] += 0.000001;
			   
			   double[] output = calc(input);
			   for(int k = 0; k < outputLayer.length; k++)
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
   	   
   	private double[] weight;//last weight is bias
   	private double[] derWeight;
   	   
   	private Activation activation;
   	private ActivationDer activationDer;
      
      public Node(Node[] inLayer, Activation activate, ActivationDer activateDer)//set values for the previous  layer and set random weights
      {
         prevLayer = inLayer;
         weight = new double[prevLayer.length + 1];
         derWeight = new double[prevLayer.length + 1];
         
         for(int i = 0; i < weight.length; i++)
            weight[i] = (Math.random() * 2 - 1) * 0.001;
         
         activation = activate;
         activationDer = activateDer;
      }
      
      public double getVal()
      {
         if(isVal)
            return val;
            
         preAct = 0;
         
         for(int i = 0; i < prevLayer.length; i++)
            preAct += prevLayer[i].getVal() * weight[i];
         
         preAct += weight[weight.length -1];
         val = activation.activate(preAct);
         
         isVal = true;
         
         return val;
      }
      
      public double getPreAct()
      {
    	  return preAct;
      }
      
      public void setVal(double input)
      {
         val = input;
         isVal = true;
      }
      
      public Node[] getPrevLayer()
      {
         return prevLayer;
      }
      
      public double getDer() 
      {
    	  return der;   
      }
      
      public void setDer(double der)
      {
    	 this.der = der;
         
         for(int i = 0; i < prevLayer.length; i++)
            derWeight[i] += momentum * prevLayer[i].getVal() * activationDer.activateDer(val, preAct) * der;
            
         derWeight[derWeight.length - 1] += momentum * activationDer.activateDer(val, preAct) * der;
         
         isDer = true;
      }
      
      public double[] getWeight()
      {
         return weight;
      }
      
      public double[] getDerWeight()
      {
         return derWeight;
      }
      
      public void setActivation(Activation activate, ActivationDer activateDer)
      {
    	  activation = activate;
    	  
          activationDer = activateDer;
      }
      
      public void saveWeightData(List<double[]> weightData, boolean recur, Node[] inputLayer)
      {
   	   if(recur && !prevLayer.equals(inputLayer))
   	   {
   		   for(int i = 0; i < prevLayer.length; i++)
   			   prevLayer[i].saveWeightData(weightData, i == 0, inputLayer);
   	   }
   	   
   	   weightData.add(weight);
      }
      
      public void saveDerWeightData(List<double[]> derWeightData, boolean recur, Node[] inputLayer)
      {
   	   if(recur && !prevLayer.equals(inputLayer))
   	   {
   		   for(int i = 0; i < prevLayer.length; i++)
   			   prevLayer[i].saveDerWeightData(derWeightData, i == 0, inputLayer);
   	   }
   	   
   	   derWeightData.add(derWeight);
      }
      
      //assumes prevLayers weights have been calculated. Calculates the derivatives for the nodes which are two layers behind it.
      // in doing so sets up the assumption for backProp on the next layer up. Stops recursion when the layer two layers behind it is the front of the network
      public void backProp(Node[] currLayer)
      {  
    	 for(int i = 0; i < prevLayer.length; i++)
         {
            double sumDer = 0;
            
            for(Node curNode: currLayer)//sum the derivatives of node
           		 sumDer += activationDer.activateDer(curNode.getVal(), curNode.getPreAct())  * curNode.getWeight()[i] * curNode.getDer();
              
            prevLayer[i].setDer(sumDer);
         }
         
         if(!prevLayer.equals(inputLayer))
        	 prevLayer[0].backProp(prevLayer);//recur on layer down
      }
      
      public void updateWeight()
      {
         if(isDer == false)
            return;
            
         for(int i = 0; i < prevLayer.length; i++)
         {
            weight[i] -= normalize(derWeight[i] * rate);
            prevLayer[i].updateWeight();
            derWeight[i] *= 1.0 - momentum;
         }
         
         weight[weight.length - 1] -= normalize(derWeight[derWeight.length - 1] * rate);
         derWeight[derWeight.length - 1] *= 1.0 - momentum;
         isDer = false;
      }
      
      public double normalize(double der)
      {
    	  if(der < 0.1 && der > -0.1)
    		  return der;
    	  if(der > 0)
    		  return 0.1;
    	  return -0.1;
      }
      
      public void clear(Node[] currLayer)
      {
         if(!isVal)
            return;
         
         val = 0;
         isVal = false;
         
         if(!currLayer.equals(inputLayer))
         {
            for(int i = 0; i < prevLayer.length; i++)
               prevLayer[i].clear(prevLayer);
         }
      }
      
      public void clearDer()
      {
    	  for(Node prevNode: prevLayer)
    		  prevNode.setDer(0.0);
    	  
    	  if(!prevLayer.equals(inputLayer))
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