# etl/03_build_analytics.py
# this script reads data/unified.csv and writes simple analytics csv files

import os  # for making folders
import pandas as pd  # for data tables


def make_output_folder(folder):
    # this function makes sure the analytics folder exists
    os.makedirs(folder, exist_ok=True)  # creates folder if missing


def avg_delivery(df, out_path):
    # this function computes average delivery time per customer
    d = (
        df[df["delivery_time_days"].notna()]  # keep rows with delivery days
        .groupby("customer_unique_id")["delivery_time_days"]  # group by customer
        .agg(["mean", "count"])  # compute mean and count
        .reset_index()  # reset index to normal
        .rename(columns={"mean": "avg_days", "count": "n"})  # rename columns
    )
    d.to_csv(out_path, index=False)  # save to csv
    print("wrote:", out_path)  # print path


def payment_dist(df, out_path):
    # this function computes distribution of payment types
    d = (
        df.groupby("payment_type")  # group by payment type
        .size()  # count rows
        .reset_index(name="n")  # make count a column
    )
    total = d["n"].sum()  # compute total
    d["pct"] = (100.0 * d["n"] / total).round(2)  # compute percentage
    d = d.sort_values("n", ascending=False)  # sort by count
    d.to_csv(out_path, index=False)  # save to csv
    print("wrote:", out_path)  # print path


def review_trend(df, out_path):
    # this function computes average review score by month
    d = (
        df.assign(month=pd.to_datetime(df["purchase_date"]).dt.to_period("M").dt.to_timestamp())  # extract month
        .groupby("month")["review_score"]  # group by month
        .mean()  # average score
        .reset_index(name="avg_score")  # reset index
    )
    d.to_csv(out_path, index=False)  # save to csv
    print("wrote:", out_path)  # print path


def rfm_snapshot(df, out_path, reference_date="2018-09-01"):
    # this function computes recency frequency monetary snapshot
    d = (
        df.groupby("customer_unique_id")  # group by customer
        .agg(
            last_order=("purchase_date", "max"),  # last order date
            frequency=("order_id", "count"),  # how many orders
            monetary=("total_payment", "sum"),  # total money spent
        )
        .reset_index()  # reset index
    )
    ref_date = pd.to_datetime(reference_date)  # convert reference date
    d["last_order"] = pd.to_datetime(d["last_order"])  # ensure datetime
    d["recency_days"] = (ref_date - d["last_order"]).dt.days  # days since last order
    d = d.drop(columns=["last_order"])  # drop last_order col
    d.to_csv(out_path, index=False)  # save to csv
    print("wrote:", out_path)  # print path


def main(
    unified_csv="data/olist_cleaned.csv",  # path to unified csv
    analytics_dir="analytics"  # folder for analytics csv files
):
    # this function runs all analytics jobs

    make_output_folder(analytics_dir)  # ensure folder exists

    try:
        df = pd.read_csv(unified_csv, parse_dates=["purchase_date"])  # read csv with date
    except FileNotFoundError:
        print("file not found:", unified_csv)  # print missing file
        return
    except Exception as e:
        print("failed to read csv:", e)  # print other error
        return

    avg_delivery(df, os.path.join(analytics_dir, "avg_delivery.csv"))  # make avg delivery file
    payment_dist(df, os.path.join(analytics_dir, "payment_dist.csv"))  # make payment dist file
    review_trend(df, os.path.join(analytics_dir, "review_trend.csv"))  # make review trend file
    rfm_snapshot(df, os.path.join(analytics_dir, "rfm.csv"))  # make rfm file

    print("done making analytics")  # final message


if __name__ == "__main__":
    # this is the main entry point
    main()  # run the function with defaults
