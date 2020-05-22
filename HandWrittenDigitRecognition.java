import java.util.*;
import java.io.*;
public class HandWrittenDigitRecognition 
{
	public static void main(String[] args) throws FileNotFoundException
	{
		int[] labels = MnistReader.getLabels("train-labels.idx1-ubyte");//insert location of this file here
		List<int[][]> imageinput = MnistReader.getImages("train-images.idx3-ubyte");//insert location of this file here
		List<double[]> images = convert3dto2d(imageinput);
		
		int[] labelstest = MnistReader.getLabels("t10k-labels.idx1-ubyte");//insert location of this file here
		List<int[][]> imageinputtest = MnistReader.getImages("t10k-images.idx3-ubyte");//insert location of this file here
		List<double[]> imagestest = convert3dto2d(imageinputtest);
		
		int[] dim = {784, 20, 15, 10};
		NeuralNetwork nn = new NeuralNetwork(dim, (x) -> (x > 0) ? x : 0, (x, y) -> (y > 0) ? 1 : 0);//replace dim with a file name and it will start from the network saved
		nn.setOutputActivation((x) -> (1 / (Math.exp(x * -1) + 1)), (x, y) -> x - x * x);
		nn.setMomentum(0.1);
		int batchSize = 10;
		
		for(int counter = 0; counter >= 0; counter++)
		{	
			//gets random samples from the larger sample
			double[][] imageBatch = new double[batchSize][images.get(0).length];
			int[] labelBatch = new int[batchSize];
			
			for(int i = 0; i < imageBatch.length; i++)
			{
				int pos = i * images.size() / imageBatch.length + (int)(Math.random() * images.size()/imageBatch.length);
				imageBatch[i] = images.get(pos);
				labelBatch[i] = labels[pos];
			}
			
			//backprop and calc error
			double error = 0;
				
			for(int i = 0; i < imageBatch.length; i++)
			{
				double[] newlabel = new double[10];
				newlabel[labelBatch[i]] = 1.0;
				error += nn.backProp(imageBatch[i], newlabel);
			}
			
			nn.saveNetwork("C:\\Users\\Agi\\eclipse-workspace\\Neural Network\\src\\digitRecogSaves\\start.txt");//insert file location for saving network
			
			System.out.printf("ER:  %.5f  GM:  %.5f  WM:  %.5f %n", 
								error / batchSize, 
								getL2(nn.getDerWeightData()), 
								getL2(nn.getWeightData()));
			
			nn.setRate(0.001);
			
			nn.updateWeight();
			
			if(false)//make true to see test score
			{
				int correct = 0;
				
				for(int i = 0; i < labelstest.length; i++)
				{
					if(labelstest[i] == indexMax(nn.calc(imagestest.get(i))))
						correct++;
				}
				
				System.out.println(correct);
			}
		}
	}
	
	/*
	 * Returns the index of the largets entry
	 */
	public static int indexMax(double[] input)
	{
	    int index = 0;
		
	    for(int i = 0; i < input.length; i++)
		index = (input[i] > input[index]) ? i : index;
		
	    return index;
	}
	
	/*
	 * Turn list of 2D array to list of 1D array
	 */
	public static List<double[]> convert3dto2d(List<int[][]> input)
	{
		List<double[]> output = new ArrayList<double[]>();
		
		for(int[][] i: input)
		{
			double[] cur = new double[i.length * i[0].length];
			for(int j = 0; j < i.length; j++)
			{
				for(int k = 0; k < i[0].length; k++)
				{
					cur[j*i[0].length + k] = i[j][k] / 255.0 - 0.5;
				}
			}
			output.add(cur);
		}
		
		return output;
	}
	
	public static double getL2(List<double[]> input)
	{
		double sum = 0;
		
		for(double[] array: input)
		{
			for(int i = 0; i < array.length; i++)
				sum += array[i] * array[i];
		}
		
		return Math.sqrt(sum);
	}
}
