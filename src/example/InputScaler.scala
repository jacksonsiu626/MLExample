package example

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler

object InputScaler {

  def main(args: Array[String]): Unit = {}
	// We need to use spark-submit command to run this program
	val conf = new SparkConf().setAppName("Input Scaler Example");
	val sc = new SparkContext(conf);
	
	// Load training data in LIBSVM format.
	val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
	
	/**
	  * Set up a StandardScaler called scaler1.
	  * This scaler will scale the features against the standard deviation 
	  * but not the mean (default behavior).
	  * The fit() function scans all data once, 
	  * calculates the global information (mean and variance),
	  * and stores them in internal states.
	  */
	val scaler1 = new StandardScaler().fit(data.map(labeledPoint => labeledPoint.features))
	/**
	  * Set up a StandardScaler called scaler2.
	  * This scaler will scale the features against both the standard deviation and the mean.
	  */
	val scaler2 = new StandardScaler(withMean = true, withStd = true).fit(data.map(labeledPoint => labeledPoint.features))
	
	// Use scaler1 to scale features of data
	// and store the result in data1.
	// The transform() function turns an unscaled feature vector into a scaled one.
	// data1's features will have unit variance. (variance = 1)
	val data1 = data.map(labeledPoint => new LabeledPoint(labeledPoint.label, scaler1.transform(labeledPoint.features)))
	
	// Use scaler2 to scale features of data
	// and store the result in data2.
	// Without converting the features into dense vectors, transformation with zero mean will raise
	// exception on sparse vector.
	// data2's features will be unit variance and zero mean.
	val data2 = data.map(labeledPoint => LabeledPoint(labeledPoint.label, scaler2.transform(Vectors.dense(labeledPoint.features.toArray))))
			
}