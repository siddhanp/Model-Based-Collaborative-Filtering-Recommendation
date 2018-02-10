import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import java.io._

object Siddhant_Patil_Task1 {
  def main(args: Array[String]): Unit = {

    val t1 = System.nanoTime
    val conf = new SparkConf().setAppName("Siddhant_Patil_Task1").setMaster("local[2]").set("spark.executor.memory","1g")
    val sc = new SparkContext(conf)


    // LOAD TRAINING DATA FROM CSV FILE.
    val train_data2 = sc.textFile(args(0))
    val train_data = train_data2.map { line =>
      val fields = line.split(",")
      ((fields(0), fields(1)), fields(2))
    }
    val first_row = train_data.first()
    val temp_train = train_data.filter(row => row != first_row)


    // LOAD TESTING DATA FROM CSV FILE.
    val test_data = sc.textFile(args(1)).map { line =>
      var fields = line.split(",")
      ((fields(0), fields(1)), 0)
    }
    val first_row1 = test_data.first()
    val test = test_data.filter(row => row != first_row1)


    // REMOVE TESTING DATA FROM TRAINING SET.
    val _train = temp_train.subtractByKey(test)
    val _new = temp_train.subtractByKey(_train)


    // GENERATE IN RATINGS FORMAT.
    val final_train = _train.map { case((x, y), z) =>
      Rating(x.toInt, y.toInt, z.toDouble)
    }


    // CONSTRUCT LOOKUP TABLE FOR MISSING VALUES.
    var predict_missing = _train.map { case((x, y), z) =>
      (x.toInt, z.toDouble)
    }
    val temp1 = predict_missing.groupByKey.mapValues(_.toList)
    val lookup = temp1.map { case(x, y) =>
      (x, y.sum/y.length)
    }


    // TRAIN THE RECOMMENDATION MODEL.
    val model = ALS.train(final_train, 10, 10, 0.01)


    // OBTAIN PREDICTIONS.
    val final_test = test.map { case((x, y), z) =>
      (x.toInt, y.toInt)
    }
    val predictions2 = model.predict(final_test).map { case Rating(x, y, z) => if (z<0) ((x, y), 0.00000) else if (z>5) ((x, y), 5.00000) else ((x, y), z)
    }


    // GET MISSING VALUES
    val final_test2 = test.map { case((x, y), z) =>
      ((x.toInt, y.toInt),0.00000)
    }
    val temp3 = final_test2.subtractByKey(predictions2).map { case((x, y), z) =>
      (x, y)
    }
    val missing = temp3.join(lookup).map { case(x, (y, z)) => ((x, y), z)
    }
    val predictions = predictions2.union(missing)
    val predictions_toPrint = predictions.map { case((x, y), z) => (x, y, z)
    }


    // CALCULATE ERROR.
    val realAndPredicted = _new.map { case ((x, y), z) =>
      ((x.toInt, y.toInt), z.toDouble)
    }.join(predictions)

    val to_print = realAndPredicted.map { case ((x, y), (r1, r2)) =>
      val err = Math.abs(r1 - r2)
      (x, y, err)
    }


    // PRINT TO OUTPUT FILE.
    val pw = new PrintWriter(new File("Siddhant_Patil_result_task1.txt"))
    pw.write("UserId,MovieId,Pred_rating\n")
    val print_ready = predictions_toPrint.sortBy(r => (r._1, r._2, r._3)).collect.toList
    for(line <- print_ready){
      pw.write(line._1+","+line._2+","+line._3+"\n")
    }
    pw.close()


    // ACCURACY INFORMATION.
    val difference = to_print.map { case (x, y, z) =>
      z
    }
    val diff = difference.collect.toList
    var c1 = 0
    var c2 = 0
    var c3 = 0
    var c4 = 0
    var c5 = 0
    for (e <- diff){
      if(e >=0 && e <1)
        c1 = c1 + 1
      else if(e >=1 && e<2)
        c2 = c2 + 1
      else if(e >= 2 && e<3)
        c3 = c3 + 1
      else if(e >= 3 && e<4)
        c4 = c4 + 1
      else if(e >= 4)
        c5 = c5 + 1
    }


    // CALCULATE RMSE.
    val MSE = to_print.map { case (x, y, z) =>
      z * z
    }.mean()
    val RMSE = Math.sqrt(MSE)


    // OUTPUT.
    println(">=0 and <1: "+c1)
    println(">=1 and <2: "+c2)
    println(">=2 and <3: "+c3)
    println(">=3 and <4: "+c4)
    println(">=4 "+c5)
    println("RMSE = "+RMSE)


    // END CLOCK.
    val duration = (System.nanoTime - t1) / 1e9d
    println("The total execution time taken is "+BigDecimal(duration).setScale(4, BigDecimal.RoundingMode.HALF_UP).toDouble+" sec.")
    sc.stop()
  }
}