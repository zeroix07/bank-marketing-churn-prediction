# How to run this project

- clone this repo using (git clone https://github.com/zeroix07/bank-marketing-churn-prediction.git) or Download Zip

- create virtual environment using this command:

python3 or python -m venv <your virtual environmet> (example : python3 -m venv .venv)

- activate the virtual environment

Windows (use CMD):
<your virtual environment>\Scripts\activate.bat (example : .venv\Scripts\activate.bat)

or

Mac/Linux

source .venv/bin/activate

- Go to main.ipynb and run all the cell

# Data Understanding

1 - age (numeric)

2 - job : type of job (categorical: "admin.","unknown","unemployed", "management","housemaid","entrepreneur","student", "blue-collar", "self-employed","retired","technician","services") 
 
3 - marital : marital status (categorical: "married","divorced","single"; 
 
### note: "divorced" means divorced or widowed)

4 - education (categorical: "unknown","secondary","primary","tertiary")
 
5 - default: has credit in default? (binary: "yes","no")
 
6 - balance: average yearly balance, in euros (numeric) 
 
7 - housing: has housing loan? (binary: "yes","no")
 
8 - loan: has personal loan? (binary: "yes","no")

### related with the last contact of the current campaign:
 
9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
 
10 - day: last contact day of the month (numeric)
 
11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
 
12 - duration: last contact duration, in seconds (numeric) 
 
### other attributes:
 
13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
 
14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
 
15 - previous: number of contacts performed before this campaign and for this client (numeric)
 
16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

Output variable (desired target):
 
17 - y - has the client subscribed a term deposit? (binary: "yes","no")