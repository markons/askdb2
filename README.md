# askdb2
___Python code to get data from Db2 database tables, using AI framework___

This project demonstrates how to connect an IBM Db2 database to a large language model (LLM) using LangChain and OpenAI—allowing you to run natural language queries and retrieve full SQL results with rich formatting.

## 📌 Features
- Connect to IBM Db2 using SQLAlchemy and ibm_db
- Patch LangChain's SQLDatabase to avoid row truncation
- Use OpenAI's GPT-4 via LangChain's ChatOpenAI wrapper
- Generate SQL queries from natural language prompts
- Display full query results in a clean tabulated format
- Includes Windows-specific DLL path setup for DB2 clients

## 📌 Components Used

 
| Module             | Purpose                                                       |
|--------------------|---------------------------------------------------------------|
| ibm_db, ibm_db_dbi | Low-level<br> and DB-API 2.0 compliant interfaces for IBM Db2 |
| SQLAlchemy         | Database<br> abstraction and engine creation                  |
| LangChain          | Framework<br> for LLM-powered applications                    |
| ChatOpenAI         | LangChain<br> wrapper for OpenAI chat models                  |
| tabulate           | Pretty-printing<br> SQL results in table format               |
 
## 📌 Setup Instructions

1. Install Dependencies
pip install ibm_db sqlalchemy langchain langchain-community langchain-openai tabulate

2. Configure Environment
Set your DB2 credentials and OpenAI API key:
DB_USERNAME = "your-user"
DB_PASSWORD = "your-pw"
DB_HOST     = "localhost"
DB_PORT     = "25000"
DB_NAME     = "SAMPLE"
export OPENAI_API_KEY=your-openai-key

3. Windows DLL Path Setup (Optional)
If you're on Windows, the script automatically adds common DB2 client paths to the DLL search path. You can customize these paths if needed.

## 📌 Usage

Run the script:
- python db2_langchain_agent.py
- Edit the query_input variable to test different natural language queries, e.g.:
query_input += "List all departments and the number of employees in each."

## 📌 The agent will:

- Interpret your query
- Generate and execute SQL
- Print tabulated results
- Provide LLM commentary

## 📌 Example Output

**SQL Query:**
SELECT department_name, COUNT(*) FROM employees GROUP BY department_name;```


| department_name | count |
|-----------------|-------|
| Sales           | 12    |
| Engineering     | 25    |
| HR              | 5     |


=== LLM Commentary (free text) ===
There are 3 departments with varying employee counts. Engineering has the most.

## 📌Advanced Notes
- SQLDatabaseNoLimit: Prevents LangChain from injecting LIMIT clauses
- ChatOpenAINoStop: Removes unsupported stop parameter from OpenAI calls
- AgentExecutor: Configured with return_intermediate_steps=True to extract raw SQL

## 📌 Caveats
- Requires a working IBM Db2 instance and valid credentials
- Assumes GPT-4 access via OpenAI

 


