import pandas as pd

def export_results():

    data={
        "Metric":["Accuracy","Precision","Recall"],
        "Value":[0.91,0.90,0.92]
    }

    df=pd.DataFrame(data)

    df.to_csv("model_results.csv",index=False)

    print("Results exported to CSV for Power BI")