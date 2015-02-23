package example

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint

object DataType {

  def main(args: Array[String]): Unit = {}
    // We need to use spark-submit command to run this program
    val conf = new SparkConf().setAppName("Data Type and Data File Example");
    val sc = new SparkContext(conf);
    //val sc = new SparkContext("local", "DataType")
    
    /********************************************
     * Vector and Vectors                       *
     ********************************************/
	val dv: Vector = Vectors.dense(56.0, 0.0, 78.0)
	val sv1: Vector = Vectors.sparse(3, Array(0, 2), Array(56.0, 78.0))
	val sv2: Vector = Vectors.sparse(3, Seq((0, 56.0), (2, 78.0)))
	
	println("dv = " + dv.toArray.mkString(" "))
	println("sv1 = " + sv1.toArray.mkString(" "))
	println("sv2 = " + sv2.toArray.mkString(" "))
	
	println("-"*50)
	
    /********************************************
     * Parsing LIBSVM file                      *
     ********************************************/
	val examples: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
	val point1 = examples.take(1)(0)
	println("point1.label = " + point1.label)
	println("Point1.features = " + point1.features.toArray.mkString(" "))
	
	println("-"*50)
	
    /********************************************
     * Parsing text file                        *
     ********************************************/	
	val data = sc.textFile("data/mllib/ridge-data/lpsa.data")
	val parsedData = data.map { line =>
		val parts = line.split(',')
		LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
	}.cache()
	val point2 = parsedData.take(1)(0)
	println("point2.label = " + point2.label)
	println("Point2.features = " + point2.features.toArray.mkString(" "))
	
	println("-"*50)
}