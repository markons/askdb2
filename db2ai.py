# -*- coding: utf-8 -*-
"""
Minimal working DB2 + LangChain example.
- Create a DB2 SQLAlchemy engine
- Wrap it in SQLDatabase (patched to avoid LangChain row truncation)
- Create SQL agent (create_sql_agent) and AgentExecutor with return_intermediate_steps=True
- Run a natural-language query, and print nicely formatted full SQL results

Components:

# --- IBM DB2 Python Modules ---
# ibm_db and ibm_db_dbi are Python libraries for connecting to IBM Db2 databases.
# - ibm_db provides a low-level API for executing SQL statements and managing connections.
# - ibm_db_dbi offers a DB-API 2.0 compliant interface, making it easier to use with Python database tools and frameworks.

# --- SQLAlchemy ---
# SQLAlchemy is a powerful Python toolkit for working with databases.
# - It provides an Object Relational Mapper (ORM) and a flexible SQL expression language.
# - In this script, it’s used to create a database engine, inspect schemas/tables, and run SQL queries in a database-agnostic way.

# --- Langchain ---
# Langchain is a framework for building applications powered by large language models (LLMs).
# - It allows you to connect LLMs (like OpenAI’s GPT models) to external data sources, including SQL databases.
# - Here, Langchain is used to interpret natural language queries, generate SQL statements, execute them, and return results in plain language.

# --- ChatOpenAI ---
# ChatOpenAI is a Langchain wrapper for OpenAI’s chat models (like GPT-3.5 or GPT-4).
# - It lets you send prompts and receive responses from the LLM, which can be used for tasks like generating SQL queries from user questions.

# --- SQLDatabaseChain --- deprecated
# SQLDatabaseChain is a Langchain component that connects an LLM to a SQL database.
# - It takes a user question, uses the LLM to generate a SQL query, runs the query, and returns the answer.
# - Useful for building natural language interfaces to databases.

"""

import os
import sys
import re
import ast
from tabulate import tabulate

from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentExecutor, AgentType
from langchain_openai import ChatOpenAI

# ---------------------------
# Add IBM DB2 directories to
# DLL search path
# (Windows specific workaround)
# ---------------------------
# Common IBM DB2 client paths
# are searched:
# ---------------------------
possible_paths = [
    r"C:\Program Files\IBM\SQLLIB\BIN",
    r"C:\Program Files (x86)\IBM\SQLLIB\BIN",
    r"C:\IBM\SQLLIB\BIN",
    # Add your specific path here if different or if using a different driver location
]

for path in possible_paths:
    if os.path.exists(path):
        print(f"Adding to DLL search path: {path}")
        os.add_dll_directory(path)

# ---------------------------
# Configuration (edit these)
# ---------------------------
DB_USERNAME = "your-user"
DB_PASSWORD = "your-pw"
DB_HOST = "localhost"
DB_PORT = "25000"
DB_NAME = "SAMPLE"

# SQLAlchemy URI for DB2
DB2_URI = f"db2+ibm_db://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# ---------------------------
# Helper / Patch classes
# ---------------------------
class SQLDatabaseNoLimit(SQLDatabase):
    """Subclass that prevents LangChain from injecting LIMIT into queries."""

    def run(self, command: str, fetch: str = "all", **kwargs):
        return super().run(command, fetch=fetch, **kwargs)


class ChatOpenAINoStop(ChatOpenAI):
    """Subclass that removes unsupported `stop` parameter."""

    def _generate(self, *args, **kwargs):
        kwargs.pop("stop", None)
        return super()._generate(*args, **kwargs)


# ---------------------------
# Utilities to format results
# ---------------------------
def extract_sql_result_from_string(log_text: str):
    if not isinstance(log_text, str):
        log_text = str(log_text)
    m = re.search(r"(\[\s*\([^\]]+\)\s*\])", log_text, re.DOTALL)
    if not m:
        return []
    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return []


def parse_select_columns(sql_query: str, raw_result: list):
    if not raw_result:
        return []
    m = re.search(r"SELECT\s+(.*?)\s+FROM\s", sql_query, re.IGNORECASE | re.DOTALL)
    if not m:
        return [f"col{i+1}" for i in range(len(raw_result[0]))]
    cols = m.group(1).strip()
    if cols == "*" or cols.endswith(".*"):
        return [f"col{i+1}" for i in range(len(raw_result[0]))]
    names = []
    for c in cols.split(","):
        c = c.strip()
        if " AS " in c.upper():
            names.append(c.split()[-1])
        elif "." in c:
            names.append(c.split(".")[-1])
        else:
            names.append(c.split()[-1])
    return names


def format_sql_result(raw_result, column_names=None):
    if not raw_result:
        return "No rows returned."
    rows = [tuple(r) for r in raw_result]
    return tabulate(rows, headers=column_names, tablefmt="psql")


def print_agent_sql_results(result):
    inter = result.get("intermediate_steps", [])
    printed_any = False

    for step in inter:
        if not isinstance(step, (list, tuple)) or len(step) < 2:
            continue
        tool_name, tool_output = step[0], step[1]
        if "sql_db_query" not in str(tool_name):
            continue

        sql_query, rows = "", []
        s = str(tool_output)
        m = re.search(r"'query':\s*'([^']+)'", s)
        if m:
            sql_query = m.group(1)
        rows = extract_sql_result_from_string(s)

        if rows:
            cols = parse_select_columns(sql_query, rows)
            print("\n--- SQL Query Result ---")
            if sql_query:
                print("Query:", sql_query)
            print(format_sql_result(rows, cols))
            printed_any = True

    if not printed_any:
        print("No SQL intermediate results found in agent output.")


# ---------------------------
# Setup DB and LangChain
# ---------------------------
def setup_database_connection_return_db():
    try:
        engine = create_engine(DB2_URI)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1 FROM SYSIBM.SYSDUMMY1"))
        return SQLDatabaseNoLimit(engine=engine)
    except Exception as e:
        print("Database setup failed:", e)
        sys.exit(1)


def setup_langchain_components(db: SQLDatabase):
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY not set in environment.")

    llm = ChatOpenAINoStop(
        temperature=0,
        max_tokens=500,
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4",
    )

    custom_prefix = """
    When generating SQL queries, do not include LIMIT or FETCH FIRST clauses unless explicitly requested.
    Ensure the query retrieves all rows from the result set.
    """

    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        prefix=custom_prefix,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent.agent,
        tools=agent.tools,
        verbose=True,
        return_intermediate_steps=True,
    )

    return agent_executor


# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    db = setup_database_connection_return_db()
    print("DB wrapper created. Usable tables:", db.get_usable_table_names())

    agent_executor = setup_langchain_components(db)

    # Example queries (uncomment one to test)
    query_input = ""
    # query_input += "Find employees with possibly chinese-originated family names. "

    # query_input += "Select all employees whose salary is greater than 50,000."
    # query_input += " Find employees who joined in the last 30 days."
    # query_input += "Retrieve distinct department names from the employee table."
    # query_input += "Sort employees by salary in descending order."
    # query_input += "Count the number of employees in each department."
    # query_input += "Find employees whose name starts with 'A'."
    # query_input += "Retrieve top 3 highest-paid employees."
    # query_input += "Write a query to find NULL values in any table column. Take care to produce syntatically correct SQL. Reduce the results for the first 3 hits. Check value occurences in those tables." #originally: "Write a query to find NULL values in a column."
    # query_input += "Display employees who don't have a manager assigned."
    # query_input += "Difference between WHERE and HAVING? Write queries for both. Show also the results."
    # query_input += "Write a query to get employees along with their department names."
    # query_input += "Find employees who don't belong to any department."
    query_input += "List all departments and the number of employees in each."
    # query_input += "Get employees and their manager's name (self join)."
    # query_input += "Find customers who placed orders in the last 7 days."
    # query_input += "Write a query to fetch employees working in more than one project. "

    result = agent_executor.invoke({"input": query_input})

    print("\n=== Raw SQL Results (tabulated) ===")
    print_agent_sql_results(result)

    print("\n=== LLM Commentary (free text) ===")
    print(result.get("output", "(no output)"))
