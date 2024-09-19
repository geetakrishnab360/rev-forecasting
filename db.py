import sqlite3
import os
import pandas as pd
import streamlit as st


def insert_experiment_into_db(
        cohort_df, selected_features, experiment_name, user
):
    cohort_values = cohort_df[selected_features].apply(
        lambda row: "|".join(str(r) for r in row), axis=1
    )
    cohort_features = "|".join(selected_features)
    cohort_details = [cohort_features] + cohort_values.tolist()
    probabilities = [0] + cohort_df["eff_probability"].tolist()
    selected = [True] + cohort_df["selected"].tolist()
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
            selected,
        )
    )
    sql_query = f"""INSERT INTO cohorts VALUES {','.join(['(' + ','.join([
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
                probabilities real not null,
                selected boolean not null
            );
            """
    )
    conn.commit()
    conn.close()


@convert_to_df
def fetch_experiment(user, experiment):
    conn = sqlite3.connect(os.environ["DB_NAME"])
    c = conn.cursor()
    c.execute(
        f"""
              SELECT rank, columns, probabilities, selected FROM cohorts
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


def create_bu_database():
    conn = sqlite3.connect(os.environ["DB_NAME"])
    c = conn.cursor()
    c.execute(
        """
            CREATE TABLE IF NOT EXISTS bu_history(
                user varchar(100) not null,
                bu varchar(50) not null,
                ds date not null,
                y real not null
            );
            """
    )
    conn.commit()
    conn.close()


def insert_bu_into_db(all_bu_dfs, user):
    conn = sqlite3.connect(os.environ["DB_NAME"])
    c = conn.cursor()

    for i, row in all_bu_dfs.iterrows():
        sql_query = f"""
            INSERT INTO bu_history (user, bu, ds, y)
            VALUES ('{user}','{row['bu']}','{row['ds']}',{row['y']})
        """
        c.execute(sql_query)

    conn.commit()
    conn.close()

def update_bu_in_db(new_df, user):
    conn = sqlite3.connect(os.environ["DB_NAME"])
    c = conn.cursor()

    for i, row in new_df.iterrows():
        sql_query = f"""
            UPDATE bu_history
            SET y = {row['y']}
            WHERE user = '{user}' AND bu = '{row['bu']}' AND ds = '{row['ds']}'
        """
        c.execute(sql_query)

    conn.commit()
    conn.close()

def delete_bu_from_db(user):
    conn = sqlite3.connect(os.environ["DB_NAME"])
    c = conn.cursor()
    c.execute(f"DELETE FROM bu_history WHERE user = '{user}'")
    conn.commit()
    conn.close()

@st.cache_data(show_spinner=False)
@convert_to_df
def fetch_all_bu_data(user):
    conn = sqlite3.connect(os.environ["DB_NAME"])
    c = conn.cursor()
    c.execute(
        f"""
              SELECT bu,ds,y FROM bu_history
              WHERE user = '{user}'
              """
    )
    return c.fetchall(), c.description
