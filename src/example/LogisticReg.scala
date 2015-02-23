package example

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

object LogisticReg {

  def main(args: Array[String]): Unit = {
    // We need to use spark-submit command to run this program
    val conf = new SparkConf().setAppName("Logistic Regression Example");
    val sc = new SparkContext(conf);
    
    // Load training data in LIBSVM format.
	val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
	
	// Split data into training (60%) and test (40%).
	val splits = data.randomSplit(Array(0.6, 0.4), seed = System.currentTimeMillis)
	val trainingSet = splits(0).cache()
	val testSet = splits(1)
	
	// Train the model
	val numIterations = 100
	/** 
	  * Similarly to LinearRegressionModel,
	  * here LogisticRegressionModel is a pre-defined object.
	  * It provides a train() function that returns a trained LogisticRegressionModel model
	  * with default settings.
	  */
	val trainedModel = LogisticRegressionWithSGD.train(trainingSet, numIterations)
	
	// Compute predicted labels on the test set 
	val actualAndPredictedLabels = testSet.map { labeledPoint =>
	  // Similarly to LinearRegressionModel,
	  // the LogisticRegressionModel provides a predict() function
	  // that receives a feature vector and outputs a predicted label.
	  val prediction = trainedModel.predict(labeledPoint.features)
	  (prediction, labeledPoint.label)
	}
	// BinaryClassificationMetrics is a class
	// that helps you to calculate some quality measurements
	// for a binary classifier.
	val metrics = new BinaryClassificationMetrics(actualAndPredictedLabels)
	val auROC = metrics.areaUnderROC()
	
	println("Area under ROC = " + auROC)  
  }

}