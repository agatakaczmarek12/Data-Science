
"""
Created on Wed Mar 27 09:19:45 2019

@author: agatakaczmarek
"""
import pandas as pd

import mysql.connector
cnx = mysql.connector.connect(user='agatakaczmarek', password='agatakaczmarek', host='dbagatakaczmarek.cij3oz5jealx.us-east-1.rds.amazonaws.com', database='agata')

df = pd.read_sql_query("select * from Critical_serv;", cnx)

cnx = mysql.connector.connect(user='agatakaczmarek', password='agatakaczmarek', host='dbagatakaczmarek.cij3oz5jealx.us-east-1.rds.amazonaws.com', database='agata')

df1 = pd.read_sql_query("select * from Incidents;", cnx)

cnx = mysql.connector.connect(user='agatakaczmarek', password='agatakaczmarek', host='dbagatakaczmarek.cij3oz5jealx.us-east-1.rds.amazonaws.com', database='agata')

df2 = pd.read_sql_query("select * from MONTHLY_INCIDENTS_JAN ;", cnx)

