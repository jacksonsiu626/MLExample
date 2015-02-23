package example

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors


object LinearReg {

  def main(args: Array[String]): Unit = {
    // We need to use spark-submit command to run this program
    val conf = new SparkConf().setAppName("Linear Regression Example");
    val sc = new SparkContext(conf);
    
    // Load and parse the data
	val data = sc.textFile("data/mllib/ridge-data/lpsa.data")
	val parsedData = data.map { line =>
	  val parts = line.split(',')
	  LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
	}.cache()
	
	// Train the model
	val numIterations = 100
	/** 
	  * LinearRegressionWithSGD is the name of a pre-defined object.
	  * The train() function returns an object of type LinearRegressionModel
	  * that has been trained on the input data.
	  * It uses stochastic gradient descent (SGD) as the training algorithm.
	  * It uses the default model settings (e.g., no intercept).
	  */
	val trainedModel = LinearRegressionWithSGD.train(parsedData, numIterations)
	
	// Evaluate model on training examples and compute training error
	val actualAndPredictedLabels = parsedData.map { labeledPoint =>
	  // The predict() function of a model receives a feature vector,
	  // and returns a predicted label value.
	  val prediction = trainedModel.predict(labeledPoint.features)
	  (labeledPoint.label, prediction)
	}
	val MSE = actualAndPredictedLabels.map{case(v, p) => math.pow((v - p), 2)}.mean()
	println("training Mean Squared Error = " + MSE)
  }
  

}
