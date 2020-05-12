import java.util.*;
import java.io.*;
public class HandWrittenDigitRecognition 
{
	public static void main(String[] args) throws FileNotFoundException
	{
		int[] labels = MnistReader.getLabels("C:\\Users\\Agi\\eclipse-workspace\\Neural Network\\src\\train-labels.idx1-ubyte");//insert location of this file here
		List<int[][]> imageinput = MnistReader.getImages("C:\\Users\\Agi\\eclipse-workspace\\Neural Network\\src\\train-images.idx3-ubyte");//insert location of this file here
		List<double[]> images = convert3dto2d(imageinput);
		
		int[] labelstest = MnistReader.getLabels("C:\\Users\\Agi\\eclipse-workspace\\Neural Network\\src\\t10k-labels.idx1-ubyte");//insert location of this file here
		List<int[][]> imageinputtest = MnistReader.getImages("C:\\Users\\Agi\\eclipse-workspace\\Neural Network\\src\\t10k-images.idx3-ubyte");//insert location of this file here
		List<double[]> imagestest = convert3dto2d(imageinputtest);
		
		int[] dim = {784, 300, 100, 10};
		
		NeuralNetwork nn = new NeuralNetwork(dim, (x) -> (x > 0) ? x : 0, (x, y) -> (y > 0) ? 1 : 0);//replace dim with a file name and it will start from the network saved
		
		nn.outputActivation((x) -> (1 / (Math.exp(x * -1) + 1)), (x, y) -> x - x * x);
		
		nn.setMomentum(0.8);
		
		int batchSize = 50;
		
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
			
			//learning rate decay
			nn.setRate(0.1 / (1 + 2 * counter));
			
			//backprop and calc error
			double error = 0;
				
			for(int i = 0; i < imageBatch.length; i++)
			{
				double[] newlabel = new double[10];
				
				newlabel[labelBatch[i]] = 1.0;
						
				error += nn.backProp(imageBatch[i], newlabel);
			}
			
			nn.saveNetwork("C:\\Users\\Agi\\eclipse-workspace\\Neural Network\\src\\digitRecogSaves\\start.txt");//insert file location for saving network
						
			System.out.println(counter);
			System.out.println(error / batchSize);

			nn.updateWeight();
			
			if(false)//make true to see test score
			{
				int correct = 0;
				
				for(int i = 0; i < labelstest.length; i++)
				{
					if(labelstest[i] == indexvalue(nn.calc(imagestest.get(i))))
						correct++;
				}
				
				System.out.println(correct / labelstest.length * 1.0);
			}
		}
	}
	
	public static int indexvalue(double[] input)
	{
		int index = 0;
		
		for(int i = 0; i < input.length; i++)
			index = (input[i] > input[index]) ? i : index;
		
		return index;
	}
	
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
}
