import sqlite3
import os
import pandas as pd


def insert_experiment_into_db(
    cohort_df, selected_features, experiment_name, user
):
    cohort_values = cohort_df[selected_features].apply(
        lambda row: "|".join(str(r) for r in row), axis=1
    )
    cohort_features = "|".join(selected_features)
    cohort_details = [cohort_features] + cohort_values.tolist()
    probabilities = [0] + cohort_df["eff_probability"].tolist()
    num_cohorts = len(probabilities)
    # print(cohort_details)
    # print(probabilities)
    values = list(
        zip(
            [user] * (num_cohorts + 1),
            [experiment_name] * (num_cohorts + 1),
            range(1, num_cohorts + 1),
            cohort_details,
            probabilities,
        )
    )
    sql_query = f"""INSERT INTO cohorts VALUES {','.join(['(' +','.join([
        f"'{c}'" if isinstance(c, str) else str(c) for c in v]) + ')' for v in values])}"""

    # print(sql_query)
    conn = sqlite3.connect(os.environ["DB_NAME"])
    c = conn.cursor()
    c.execute(sql_query)
    conn.commit()
    conn.close()


def convert_to_df(func):
    def __wrapper__(*args):
        data, description = func(*args)
        if len(data) > 0:
            return pd.DataFrame(
                data, columns=[desc[0] for desc in description]
            )
        else:
            pd.DataFrame()

    return __wrapper__


def create_database():
    conn = sqlite3.connect(os.environ["DB_NAME"])
    c = conn.cursor()
    c.execute(
        """
            CREATE TABLE IF NOT EXISTS cohorts(
                user varchar(100) not null,
                experiment varchar(50) not null,
                rank int not null,
                columns varchar(1000) not null,
                probabilities real not null
            );
            """
    )

    # c.execute(
    #     # """
    #     #       INSERT INTO cohorts VALUES
    #     #         ("geeta","default",1,"deal_stage",0),
    #     #         ("geeta","default",2,"stage 2",0.1),
    #     #         ("geeta","default",3,"stage 3",0.2),
    #     #         ("geeta","default",4,"stage 4",0.3),
    #     #         ("geeta","default",5,"stage 5",0.5),
    #     #         ("geeta","default",6,"stage 6",0.75),
    #     #         ("geeta","default",7,"stage 7",0.9)
    #     #       """
    #     """
    #            INSERT INTO cohorts VALUES (gk,e1,1,0|0,0.1), gk,e1,1,0|0,0.1)
    #     """
    # )
    conn.commit()
    conn.close()


@convert_to_df
def fetch_experiment(user, experiment):
    conn = sqlite3.connect(os.environ["DB_NAME"])
    c = conn.cursor()
    c.execute(
        f"""
              SELECT rank, columns, probabilities FROM cohorts
              WHERE user = '{user}'
              AND experiment = '{experiment}'
              ORDER BY 1 ASC
              """
    )
    return c.fetchall(), c.description


def fetch_all_experiments(user):
    conn = sqlite3.connect(os.environ["DB_NAME"])
    c = conn.cursor()
    c.execute(
        f"""
              SELECT distinct experiment as experiment FROM cohorts
              WHERE user = '{user}'
              """
    )
    return [exp[0] for exp in c.fetchall()]
