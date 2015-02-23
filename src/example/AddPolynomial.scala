package example

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics


object AddPolynomial {

  def main(args: Array[String]): Unit = {
    // We need to use spark-submit command to run this program
    val conf = new SparkConf().setAppName("Add Polynomial Terms Example");
    val sc = new SparkContext(conf);
    
    // Let's make a toy example:
    // A two dimensional dataset with only two samples
    val point1 = new LabeledPoint(1.0, Vectors.dense(2.0, 3.0))
    val point2 = new LabeledPoint(0.0, Vectors.dense(40.0, 15.0))
    val data = sc.parallelize(Array(point1, point2))
    
    // Let's see it
    data.collect.foreach {point =>
      println(point.label + "," + point.features.toArray.mkString(" "))
    }
    println("-"*50)
    
    // Prepare a function that will be used to add polynomial terms.    
    def addTerms (inVec: Vector) = {
      val x1 = inVec.toArray(0)
      val x2 = inVec.toArray(1)
      Vectors.dense(x1, x2, x1*x1, x2*x2, x1*x2)
    }
    
    // Add polynomial terms to the data
    val extendedData = data.map { labeledPoint =>
      new LabeledPoint(labeledPoint.label, addTerms(labeledPoint.features))
    }
    
    // Let's see it
    extendedData.collect.foreach {point =>
      println(point.label + "," + point.features.toArray.mkString(" "))
    }
    
  }

}