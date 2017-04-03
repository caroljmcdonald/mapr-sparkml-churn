package example

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.feature.VectorAssembler

/*
 * Dataset schema
State	
Account length	
Area code	
International plan	
Voice mail plan	
Number vmail messages	
Total day minutes	
Total day calls	
Total day charge	
Total eve minutes	
Total eve calls	Total eve charge	
Total night minutes	
Total night calls	
Total night charge	
Total intl minutes	
Total intl calls	
Total intl charge	
Customer service calls	
Churn
 */

object Churn {

  case class Account(state: String, len: Integer, acode: String,
    intlplan: String, vplan: String, numvmail: Double,
    tdmins: Double, tdcalls: Double, tdcharge: Double,
    temins: Double, tecalls: Double, techarge: Double,
    tnmins: Double, tncalls: Double, tncharge: Double,
    timins: Double, ticalls: Double, ticharge: Double,
    numcs: Double, churn: String)
  val schema = StructType(Array(
    StructField("state", StringType, true),
    StructField("len", IntegerType, true),
    StructField("acode", StringType, true),
    StructField("intlplan", StringType, true),
    StructField("vplan", StringType, true),
    StructField("numvmail", DoubleType, true),
    StructField("tdmins", DoubleType, true),
    StructField("tdcalls", DoubleType, true),
    StructField("tdcharge", DoubleType, true),
    StructField("temins", DoubleType, true),
    StructField("tecalls", DoubleType, true),
    StructField("techarge", DoubleType, true),
    StructField("tnmins", DoubleType, true),
    StructField("tncalls", DoubleType, true),
    StructField("tncharge", DoubleType, true),
    StructField("timins", DoubleType, true),
    StructField("ticalls", DoubleType, true),
    StructField("ticharge", DoubleType, true),
    StructField("numcs", DoubleType, true),
    StructField("churn", StringType, true)
  ))

  def main(args: Array[String]) {

    val spark: SparkSession = SparkSession.builder().appName("churn").getOrCreate()

    import spark.implicits._

    val train: Dataset[Account] = spark.read.option("inferSchema", "false")
      .schema(schema).csv("/user/user01/data/churn-bigml-80.csv").as[Account]
    train.take(1)
    train.cache
    println(train.count)

    val test: Dataset[Account] = spark.read.option("inferSchema", "false")
      .schema(schema).csv("/user/user01/data/churn-bigml-20.csv").as[Account]
    test.take(2)
    println(test.count)
    test.cache

    train.printSchema()
    train.show
    train.createOrReplaceTempView("account")
    spark.catalog.cacheTable("account")

    train.groupBy("churn").count.show
    val fractions = Map("False" -> .17, "True" -> 1.0)
    //Here we're keeping all instances of the Churn=True class, but downsampling the Churn=False class to a fraction of 388/2278.
    val strain = train.stat.sampleBy("churn", fractions, 36L)

    strain.groupBy("churn").count.show
    val ntrain = strain.drop("state").drop("acode").drop("vplan").drop("tdcharge").drop("techarge")
    println(ntrain.count)
    ntrain.show

    val ipindexer = new StringIndexer()
      .setInputCol("intlplan")
      .setOutputCol("iplanIndex")
    val labelindexer = new StringIndexer()
      .setInputCol("churn")
      .setOutputCol("label")
    val featureCols = Array("len", "iplanIndex", "numvmail", "tdmins", "tdcalls", "temins", "tecalls", "tnmins", "tncalls", "timins", "ticalls", "numcs")

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val dTree = new DecisionTreeClassifier().setLabelCol("label")
      .setFeaturesCol("features")

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(ipindexer, labelindexer, assembler, dTree))
    // Search through decision tree's maxDepth parameter for best model
    val paramGrid = new ParamGridBuilder().addGrid(dTree.maxDepth, Array(2, 3, 4, 5, 6, 7)).build()

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    // Set up 3-fold cross validation
    val crossval = new CrossValidator().setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid).setNumFolds(3)

    val cvModel = crossval.fit(ntrain)

    val bestModel = cvModel.bestModel
    println("The Best Model and Parameters:\n--------------------")
    println(bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(3))
    bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
      .stages(3)
      .extractParamMap

    val treeModel = bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(3).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)

    val predictions = cvModel.transform(test)
    val accuracy = evaluator.evaluate(predictions)
    evaluator.explainParams()

    val predictionAndLabels = predictions.select("prediction", "label").rdd.map(x =>
      (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    println("area under the precision-recall curve: " + metrics.areaUnderPR)
    println("area under the receiver operating characteristic (ROC) curve : " + metrics.areaUnderROC)

    println(metrics.fMeasureByThreshold())

    val result = predictions.select("label", "prediction", "probability")
    result.show

    val lp = predictions.select("label", "prediction")
    val counttotal = predictions.count()
    val correct = lp.filter($"label" === $"prediction").count()
    val wrong = lp.filter(not($"label" === $"prediction")).count()
    val ratioWrong = wrong.toDouble / counttotal.toDouble
    val ratioCorrect = correct.toDouble / counttotal.toDouble
    val truep = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count() / counttotal.toDouble
    val truen = lp.filter($"prediction" === 1.0).filter($"label" === $"prediction").count() / counttotal.toDouble
    val falsep = lp.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count() / counttotal.toDouble
    val falsen = lp.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count() / counttotal.toDouble

    println("counttotal : " + counttotal)
    println("correct : " + correct)
    println("wrong: " + wrong)
    println("ratio wrong: " + ratioWrong)
    println("ratio correct: " + ratioCorrect)
    println("ratio true positive : " + truep)
    println("ratio false positive : " + falsep)
    println("ratio true negative : " + truen)
    println("ratio false negative : " + falsen)

    println("wrong: " + wrong)

    val equalp = predictions.selectExpr(
      "double(round(prediction)) as prediction", "label",
      """CASE double(round(prediction)) = label WHEN true then 1 ELSE 0 END as equal"""
    )
    equalp.show

  }
}

