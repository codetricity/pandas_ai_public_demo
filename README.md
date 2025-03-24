# Pandas AI Test

using camera360 user data

## python setup

* python must be 3.11.9 or slightly older.  will not work
with 3.12.*
* suggest using pyenv to setup python version

```text
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## install

* use requirements.txt
* the numpy, pandas and pandasai requirements are complex

## setup of security

* put OpenAI API key on config.py.  use config.py.example as reference
`OPENAI_API_KEY = "sk-proj..."`
* create `./streamlit/secrets.toml`

```text
password = "put_secret_password here"

[data]
gdrive_file_id = "put google sheets file id here"
```

## run

`streamlit run app.py`
