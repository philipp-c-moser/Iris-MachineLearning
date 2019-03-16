using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MachineLearning
{
    class Program
    {

        // Data-Structure-Definition
        public class IrisData
        {
            [LoadColumn(0)] public float SepalLength;
            [LoadColumn(1)] public float SepalWidth;
            [LoadColumn(2)] public float PetalLength;
            [LoadColumn(3)] public float PetalWidth;
            [LoadColumn(4)] public float Label;
        }

        // Result returned from prediction operations
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")] public string PredictedLabels;
        }



        static void Main(string[] args)
        {

            // MachineLearning Environment
            MLContext mlContext = new MLContext();

            IDataView trainingDataView =
                mlContext.Data.LoadFromTextFile<IrisData>(path: "iris-data.txt", hasHeader: false, separatorChar: ',');


        }
    }
}
